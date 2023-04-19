import os
import pathlib
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from .ranger import Ranger
from oncofem.mri.open_brats import models
from oncofem.mri.open_brats.dataset import get_datasets
from oncofem.mri.open_brats.dataset.batch_utils import determinist_collate
from oncofem.mri.open_brats.loss import EDiceLoss
from oncofem.mri.open_brats.models import get_norm_layer, DataAugmenter
from .utils import save_args, AverageMeter, ProgressMeter, reload_ckpt, save_checkpoint, reload_ckpt_bis, \
    count_parameters, WeightSWA, save_metrics, generate_segmentations

class TrainInput:
    def __init__(self):
        self.arch = "Unet"
        self.width = 48
        self.workers = 2
        self.start_epoch = 0
        self.epochs = 200
        self.batch_size = 1
        self.lr = 1e-4
        self.weight_decay = 0.0
        self.resume = ""
        self.devices = 0
        self.debug = True
        self.deep_sup = False
        self.no_fp16 = False
        self.seed = 16111990
        self.warm = 3
        self.val = 3
        self.fold = 0
        self.norm_layer = "group"
        self.swa = False
        self.swa_repeat = 5
        self.optim = "ranger"
        self.com = "add a comment to this run!"
        self.dropout = 0.0
        self.warm_restart = False
        self.full = False
        current_experiment_time = datetime.now().strftime('%Y%m%d_%T').replace(":", "")
        self.exp_name = f"{'debug_' if self.debug else ''}{current_experiment_time}_" \
                      f"_fold{self.fold if not self.full else 'FULL'}" \
                      f"_{self.arch}_{self.width}" \
                      f"_batch{self.batch_size}" \
                      f"_optim{self.optim}" \
                      f"_{self.optim}" \
                      f"_lr{self.lr}-wd{self.weight_decay}_epochs{self.epochs}_deepsup{self.deep_sup}" \
                      f"_{'fp16' if not self.no_fp16 else 'fp32'}" \
                      f"_warm{self.warm}_" \
                      f"_norm{self.norm_layer}{'_swa' + str(self.swa_repeat) if self.swa else ''}" \
                      f"_dropout{self.dropout}" \
                      f"_warm_restart{self.warm_restart}" \
                      f"{'_' + self.com.replace(' ', '_') if self.com else ''}"
        self.save_folder = pathlib.Path(f"./runs/{self.exp_name}")
        self.save_folder.mkdir(parents=True, exist_ok=True)
        self.seg_folder = self.save_folder / "segs"
        self.seg_folder.mkdir(parents=True, exist_ok=True)
        self.save_folder = self.save_folder.resolve()
        save_args(self)

def run_training(ip: TrainInput):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(ip.devices)
    # setup
    ngpus = torch.cuda.device_count()
    if ngpus == 0:
        raise RuntimeWarning("This will not be able to run on CPU only")

    print(f"Working with {ngpus} GPUs")
    if ip.optim.lower() == "ranger":
        ip.warm = 0

    t_writer = SummaryWriter(str(ip.save_folder))

    # Create model
    print(f"Creating {ip.arch}")

    model_maker = getattr(models, ip.arch)

    model = model_maker(4, 3, width=ip.width, deep_supervision=ip.deep_sup, norm_layer=get_norm_layer(ip.norm_layer), dropout=ip.dropout)

    print(f"total number of trainable parameters {count_parameters(model)}")

    if ip.swa:
        # Create the average model
        swa_model = model_maker(4, 3, width=ip.width, deep_supervision=ip.deep_sup, norm_layer=get_norm_layer(ip.norm_layer))
        for param in swa_model.parameters():
            param.detach_()
        swa_model = swa_model.cuda()
        swa_model_optim = WeightSWA(swa_model)

    if ngpus > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()
    print(model)
    model_file = ip.save_folder / "model.txt"
    with model_file.open("w") as f:
        print(model, file=f)

    criterion = EDiceLoss().cuda()
    metric = criterion.metric
    print(metric)

    rangered = False  # needed because LR scheduling scheme is different for this optimizer
    if ip.optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=ip.lr, weight_decay=ip.weight_decay, eps=1e-4)
    elif ip.optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=ip.lr, weight_decay=ip.weight_decay, momentum=0.9, nesterov=True)
    elif ip.optim == "adamw":
        print(f"weight decay argument will not be used. Default is 11e-2")
        optimizer = torch.optim.AdamW(model.parameters(), lr=ip.lr)
    elif ip.optim == "ranger":
        optimizer = Ranger(model.parameters(), lr=ip.lr, weight_decay=ip.weight_decay)
        rangered = True

    # optionally resume from a checkpoint
    if ip.resume:
        reload_ckpt(ip, model, optimizer)

    if ip.debug:
        ip.epochs = 2
        ip.warm = 0
        ip.val = 1

    if ip.full:
        train_dataset, bench_dataset = get_datasets(ip.seed, ip.debug, full=True)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=ip.batch_size, shuffle=True, num_workers=ip.workers, 
                                                   pin_memory=False, drop_last=True)
        bench_loader = torch.utils.data.DataLoader(bench_dataset, batch_size=1, num_workers=ip.workers)

    else:
        train_dataset, val_dataset, bench_dataset = get_datasets(ip.seed, ip.debug, fold_number=ip.fold)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=ip.batch_size, shuffle=True, 
                                                   num_workers=ip.workers, pin_memory=False, drop_last=True)

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=max(1, ip.batch_size // 2), shuffle=False, 
                                                 pin_memory=False, num_workers=ip.workers, collate_fn=determinist_collate)
        bench_loader = torch.utils.data.DataLoader(bench_dataset, batch_size=1, num_workers=ip.workers)
        print("Val dataset number of batch:", len(val_loader))

    print("Train dataset number of batch:", len(train_loader))

    # create grad scaler
    scaler = GradScaler()

    # Actual Train loop
    best = np.inf
    print("start warm-up now!")
    if ip.warm != 0:
        tot_iter_train = len(train_loader)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda cur_iter: (1 + cur_iter) / (tot_iter_train * ip.warm))

    patients_perf = []

    if not ip.resume:
        for epoch in range(ip.warm):
            ts = time.perf_counter()
            model.train()
            training_loss = step(train_loader, model, criterion, metric, ip.deep_sup, optimizer, epoch, t_writer,scaler, 
                                 scheduler, save_folder=ip.save_folder, no_fp16=ip.no_fp16, patients_perf=patients_perf)
            te = time.perf_counter()
            print(f"Train Epoch done in {te - ts} s")

            # Validate at the end of epoch every val step
            if (epoch + 1) % ip.val == 0 and not ip.full:
                model.eval()
                with torch.no_grad():
                    validation_loss = step(val_loader, model, criterion, metric, ip.deep_sup, optimizer, epoch,
                                           t_writer, save_folder=ip.save_folder, no_fp16=ip.no_fp16)

                t_writer.add_scalar(f"SummaryLoss/overfit", validation_loss - training_loss, epoch)

    if ip.warm_restart:
        print('Total number of epochs should be divisible by 30, else it will do odd things')
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 30, eta_min=1e-7)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, ip.epochs + 30 if not rangered else round(ip.epochs * 0.5))
    print("start training now!")
    if ip.swa:
        # c = 15, k=3, repeat = 5
        c, k, repeat = 30, 3, ip.swa_repeat
        epochs_done = ip.epochs
        reboot_lr = 0
        if ip.debug:
            c, k, repeat = 2, 1, 2

    for epoch in range(ip.start_epoch + ip.warm, ip.epochs + ip.warm):
        try:
            # do_epoch for one epoch
            ts = time.perf_counter()
            model.train()
            training_loss = step(train_loader, model, criterion, metric, ip.deep_sup, optimizer, epoch, t_writer,
                                 scaler, save_folder=ip.save_folder, no_fp16=ip.no_fp16, patients_perf=patients_perf)
            te = time.perf_counter()
            print(f"Train Epoch done in {te - ts} s")

            # Validate at the end of epoch every val step
            if (epoch + 1) % ip.val == 0 and not ip.full:
                model.eval()
                with torch.no_grad():
                    validation_loss = step(val_loader, model, criterion, metric, ip.deep_sup, optimizer,
                                           epoch,
                                           t_writer,
                                           save_folder=ip.save_folder,
                                           no_fp16=ip.no_fp16, patients_perf=patients_perf)

                t_writer.add_scalar(f"SummaryLoss/overfit", validation_loss - training_loss, epoch)

                if validation_loss < best:
                    best = validation_loss
                    model_dict = model.state_dict()
                    save_checkpoint(
                        dict(
                            epoch=epoch, arch=ip.arch,
                            state_dict=model_dict,
                            optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict(),
                        ),
                        save_folder=ip.save_folder, )

                ts = time.perf_counter()
                print(f"Val epoch done in {ts - te} s")

            if ip.swa:
                if (ip.epochs - epoch - c) == 0:
                    reboot_lr = optimizer.param_groups[0]['lr']

            if not rangered:
                scheduler.step()
                print("scheduler stepped!")
            else:
                if epoch / ip.epochs > 0.5:
                    scheduler.step()
                    print("scheduler stepped!")

        except KeyboardInterrupt:
            print("Stopping training loop, doing benchmark")
            break

    if ip.swa:
        swa_model_optim.update(model)
        print("SWA Model initialised!")
        for i in range(repeat):
            optimizer = torch.optim.Adam(model.parameters(), ip.lr / 2, weight_decay=ip.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, c + 10)
            for swa_epoch in range(c):
                # do_epoch for one epoch
                ts = time.perf_counter()
                model.train()
                swa_model.train()
                current_epoch = epochs_done + i * c + swa_epoch
                training_loss = step(train_loader, model, criterion, metric, ip.deep_sup, optimizer,
                                     current_epoch, t_writer,
                                     scaler, no_fp16=ip.no_fp16, patients_perf=patients_perf)
                te = time.perf_counter()
                print(f"Train Epoch done in {te - ts} s")

                t_writer.add_scalar(f"SummaryLoss/train", training_loss, current_epoch)

                # update every k epochs and val:
                print(f"cycle number: {i}, swa_epoch: {swa_epoch}, total_cycle_to_do {repeat}")
                if (swa_epoch + 1) % k == 0:
                    swa_model_optim.update(model)
                    if not ip.full:
                        model.eval()
                        swa_model.eval()
                        with torch.no_grad():
                            validation_loss = step(val_loader, model, criterion, metric, ip.deep_sup, optimizer,
                                                   current_epoch,
                                                   t_writer, save_folder=ip.save_folder, no_fp16=ip.no_fp16)
                            swa_model_loss = step(val_loader, swa_model, criterion, metric, ip.deep_sup, optimizer,
                                                  current_epoch,
                                                  t_writer, swa=True, save_folder=ip.save_folder,
                                                  no_fp16=ip.no_fp16)

                        t_writer.add_scalar(f"SummaryLoss/val", validation_loss, current_epoch)
                        t_writer.add_scalar(f"SummaryLoss/swa", swa_model_loss, current_epoch)
                        t_writer.add_scalar(f"SummaryLoss/overfit", validation_loss - training_loss, current_epoch)
                        t_writer.add_scalar(f"SummaryLoss/overfit_swa", swa_model_loss - training_loss, current_epoch)
                scheduler.step()
        epochs_added = c * repeat
        save_checkpoint(
            dict(
                epoch=ip.epochs + epochs_added, arch=ip.arch,
                state_dict=swa_model.state_dict(),
                optimizer=optimizer.state_dict()
            ),
            save_folder=ip.save_folder, )
    else:
        save_checkpoint(
            dict(
                epoch=ip.epochs, arch=ip.arch,
                state_dict=model.state_dict(),
                optimizer=optimizer.state_dict()
            ),
            save_folder=ip.save_folder, )

    try:
        df_individual_perf = pd.DataFrame.from_records(patients_perf)
        print(df_individual_perf)
        df_individual_perf.to_csv(f'{str(ip.save_folder)}/patients_indiv_perf.csv')
        reload_ckpt_bis(f'{str(ip.save_folder)}/model_best.pth.tar', model)
        generate_segmentations(bench_loader, model, t_writer, ip)
    except KeyboardInterrupt:
        print("Stopping right now!")

def step(data_loader, model, criterion: EDiceLoss, metric, deep_supervision, optimizer, epoch, writer, scaler=None,
         scheduler=None, swa=False, save_folder=None, no_fp16=False, patients_perf=None):
    # Setup
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    # TODO monitor teacher loss
    mode = "train" if model.training else "val"
    batch_per_epoch = len(data_loader)
    progress = ProgressMeter(
        batch_per_epoch,
        [batch_time, data_time, losses],
        prefix=f"{mode} Epoch: [{epoch}]")

    end = time.perf_counter()
    metrics = []
    print(f"fp 16: {not no_fp16}")
    # TODO: not recreate data_aug for each epoch...
    data_aug = DataAugmenter(p=0.8, noise_only=False, channel_shuffling=False, drop_channnel=True).cuda()

    for i, batch in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.perf_counter() - end)

        targets = batch["label"].cuda(non_blocking=True)
        inputs = batch["image"].cuda()
        patient_id = batch["patient_id"]

        with autocast(enabled=not no_fp16):
            # data augmentation step
            if mode == "train":
                inputs = data_aug(inputs)
            if deep_supervision:
                segs, deeps = model(inputs)
                if mode == "train":  # revert the data aug
                    segs, deeps = data_aug.reverse([segs, deeps])
                loss_ = torch.stack(
                    [criterion(segs, targets)] + [criterion(deep, targets) for
                                                  deep in deeps])
                print(f"main loss: {loss_}")
                loss_ = torch.mean(loss_)
            else:
                segs = model(inputs)
                if mode == "train":
                    segs = data_aug.reverse(segs)
                loss_ = criterion(segs, targets)
            if patients_perf is not None:
                patients_perf.append(
                    dict(id=patient_id[0], epoch=epoch, split=mode, loss=loss_.item())
                )

            writer.add_scalar(f"Loss/{mode}{'_swa' if swa else ''}",
                              loss_.item(),
                              global_step=batch_per_epoch * epoch + i)

            # measure accuracy and record loss_
            if not np.isnan(loss_.item()):
                losses.update(loss_.item())
            else:
                print("NaN in model loss!!")

            if not model.training:
                metric_ = metric(segs, targets)
                metrics.extend(metric_)

        # compute gradient and do SGD step
        if model.training:
            scaler.scale(loss_).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            writer.add_scalar("lr", optimizer.param_groups[0]['lr'], global_step=epoch * batch_per_epoch + i)
        if scheduler is not None:
            scheduler.step()

        # measure elapsed time
        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()
        # Display progress
        progress.display(i)

    if not model.training:
        save_metrics(epoch, metrics, swa, writer, epoch, False, save_folder)

    if mode == "train":
        writer.add_scalar(f"SummaryLoss/train", losses.avg, epoch)
    else:
        writer.add_scalar(f"SummaryLoss/val", losses.avg, epoch)

    return losses.avg
