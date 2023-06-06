"""
Definition of tumor segmentation class

Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""

from oncofem.helper.general import mkdir_if_not_exist, save_args
from oncofem.helper import constant as const
from oncofem.helper.auxillaries import count_parameters, calculate_metrics
from oncofem.mri.tumor_segmentation import models
from oncofem.mri.tumor_segmentation.dataset import get_datasets
from oncofem.mri.tumor_segmentation.utils import pad_batch1_to_compatible_size, determinist_collate, pad_single_to_compatible_size
from oncofem.mri.tumor_segmentation.models import get_norm_layer, DataAugmenter
from oncofem.mri.tumor_segmentation.tta import apply_simple_tta
from oncofem.mri.tumor_segmentation.utils import reload_ckpt_bis, reload_ckpt, WeightSWA, AverageMeter, ProgressMeter, save_metrics, save_checkpoint
from oncofem.mri.tumor_segmentation.loss import EDiceLoss
from oncofem.mri.tumor_segmentation.ranger import Ranger
import oncofem.mri

import os
import time
import pathlib
import random
from types import SimpleNamespace
import nibabel as nib

import SimpleITK as sitk
import numpy as np
import pandas as pd
import torch
import torch.optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import yaml
from torch.cuda.amp import autocast, GradScaler

class TrainParam:
    def __init__(self):
        self.data_folder = None
        self.save_folder = None
        self.input_patterns = ["_t1", "_t1ce", "_t2", "_flair"]
        self.rand_blank = False
        self.input_channel = None
        self.output_channel = const.TRAINING_OUTPUT_CHANNEL
        self.arch = const.TRAINING_ARCH
        self.width = const.TRAINING_WIDTH
        self.workers = const.TRAINING_WORKERS
        self.start_epoch = const.TRAINING_START_EPOCH
        self.epochs = const.TRAINING_EPOCHS
        self.batch_size = const.TRAINING_BATCH_SIZE
        self.lr = const.TRAINING_LR
        self.weight_decay = const.TRAINING_WEIGHT_DECAY
        self.resume = const.TRAINING_RESUME
        self.debug = const.TRAINING_DEBUG
        self.deep_sup = const.TRAINING_DEEP_SUP
        self.no_fp16 = const.TRAINING_NO_FP16
        self.warm = const.TRAINING_WARM
        self.val = const.TRAINING_VAL
        self.fold = const.TRAINING_FOLD
        self.norm_layer = const.TRAINING_NORM_LAYER
        self.swa = const.TRAINING_SWA
        self.swa_repeat = const.TRAINING_SWA_REPEAT
        self.optim = const.TRAINING_OPTIM
        self.com = const.TRAINING_COM
        self.dropout = const.TRAINING_DROPOUT
        self.warm_restart = const.TRAINING_WARM_RESTART
        self.full = const.TRAINING_FULL

class InferParam:
    def __init__(self):
        self.config = const.OPEN_BRATS2020_DEFAULT_WEIGHTS_DIR
        self.input_patterns = ["_t1", "_t1ce", "_t2", "_flair"]
        self.input_data = None
        self.normalisation = "minmax"
        self.output_path = None
        self.on = "train"
        self.input = None  # former on
        self.tta = const.OPEN_BRATS2020_TTA

class TumorSegmentation:

    def __init__(self, mri):
        self.mri = mri
        self.study_dir = mri.study_dir
        self.state = mri.state
        self.dir = mri.study_dir + const.DER_DIR + mri.state.subject + os.sep + str(mri.state.date) + os.sep + const.TUMOR_SEGMENTATION_PATH
        mkdir_if_not_exist(self.dir)

        self.devices = const.OPEN_BRATS2020_DEVICES
        self.seed = const.OPEN_BRATS2020_SEED

        self.train_param = TrainParam()
        self.infer_param = InferParam()

    def run_training(self):
        """ 
        The main training function.

        Only works for single node (be it single or multi-GPU)
        """
        # setup
        ngpus = torch.cuda.device_count()
        if ngpus == 0:
            raise RuntimeWarning("This will not be able to run on CPU only")

        print("Working with " + str(ngpus) + " GPUs")
        if self.train_param.optim.lower() == "ranger":
            self.train_param.warm = 0

        mkdir_if_not_exist(self.train_param.save_folder)
        self.train_param.seg_folder = self.train_param.save_folder + os.sep + "segs"
        mkdir_if_not_exist(self.train_param.seg_folder)
        save_args(self.train_param)
        t_writer = SummaryWriter(str(self.train_param.save_folder))

        # Create model
        print("Creating " + str(self.train_param.arch))
        self.train_param.input_channel = len(self.train_param.input_patterns)
        model_maker = getattr(models, self.train_param.arch)

        model = model_maker(self.train_param.input_channel, self.train_param.output_channel,
                            width=self.train_param.width, deep_supervision=self.train_param.deep_sup,
                            norm_layer=get_norm_layer(self.train_param.norm_layer), dropout=self.train_param.dropout)

        print("total number of trainable parameters " + str(count_parameters(model)))

        if self.train_param.swa:
            # Create the average model
            swa_model = model_maker(self.train_param.input_channel, self.train_param.output_channel,
                                    width=self.train_param.width, deep_supervision=self.train_param.deep_sup,
                                    norm_layer=get_norm_layer(self.train_param.norm_layer))
            for param in swa_model.parameters():
                param.detach_()
            swa_model = swa_model.cuda()
            swa_model_optim = WeightSWA(swa_model)

        if ngpus > 1:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda()
        print(model)
        model_file = self.train_param.save_folder + os.sep + "model.txt"
        with open(model_file, "w") as f:
            print(model, file=f)

        criterion = EDiceLoss().cuda()
        metric = criterion.metric
        print(metric)

        rangered = False  # needed because LR scheduling scheme is different for this optimizer
        if self.train_param.optim == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=self.train_param.lr, 
                                         weight_decay=self.train_param.weight_decay, eps=1e-4)

        elif self.train_param.optim == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=self.train_param.lr, 
                                        weight_decay=self.train_param.weight_decay, momentum=0.9, nesterov=True)
        elif self.train_param.optim == "adamw":
            print("weight decay argument will not be used. Default is 11e-2")
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.train_param.lr)

        elif self.train_param.optim == "ranger":
            optimizer = Ranger(model.parameters(), lr=self.train_param.lr, weight_decay=self.train_param.weight_decay)
            rangered = True

        # optionally resume from a checkpoint
        if self.train_param.resume:
            reload_ckpt(self.train_param, model, optimizer)

        if self.train_param.debug:
            self.train_param.epochs = 2
            self.train_param.warm = 0
            self.train_param.val = 1

        if self.train_param.full:
            train_dataset, bench_dataset = get_datasets(self.train_param.data_folder, self.train_param.input_patterns,
                                                        self.train_param.seed, self.train_param.debug, full=True)

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.train_param.batch_size, 
                                                       shuffle=True, num_workers=self.train_param.workers, 
                                                       pin_memory=False, drop_last=True)

            bench_loader = torch.utils.data.DataLoader(bench_dataset, batch_size=1, num_workers=self.train_param.workers)

        else:
            train_dataset, val_dataset, bench_dataset = get_datasets(self.train_param.data_folder, self.train_param.input_patterns,
                                                                     self.seed, self.train_param.debug, fold_number=self.train_param.fold)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.train_param.batch_size, 
                                                       shuffle=True, num_workers=self.train_param.workers, 
                                                       pin_memory=False, drop_last=True)

            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=max(1, self.train_param.batch_size // 2), 
                                                     shuffle=False, pin_memory=False, 
                                                     num_workers=self.train_param.workers, collate_fn=determinist_collate)

            bench_loader = torch.utils.data.DataLoader(bench_dataset, batch_size=1, num_workers=self.train_param.workers)
            print("Val dataset number of batch:", len(val_loader))

        print("Train dataset number of batch:", len(train_loader))
        # create grad scaler
        scaler = GradScaler()

        # Actual Train loop
        best = np.inf
        print("start warm-up now!")
        if self.train_param.warm != 0:
            tot_iter_train = len(train_loader)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda cur_iter: (1 + cur_iter)/(tot_iter_train * self.train_param.warm))

        patients_perf = []

        if not self.train_param.resume:
            for epoch in range(self.train_param.warm):
                ts = time.perf_counter()
                model.train()
                training_loss = self.step(train_loader, model, criterion, metric, self.train_param.deep_sup, optimizer, 
                                          epoch, t_writer, scaler, scheduler, save_folder=self.train_param.save_folder,
                                          no_fp16=self.train_param.no_fp16, patients_perf=patients_perf)
                te = time.perf_counter()
                print("Train Epoch done in " + str(te - ts) + " s")

                # Validate at the end of epoch every val step
                if (epoch + 1) % self.train_param.val == 0 and not self.train_param.full:
                    model.eval()
                    with torch.no_grad():
                        validation_loss = self.step(val_loader, model, criterion, metric, self.train_param.deep_sup, optimizer, epoch,
                                                    t_writer, save_folder=self.train_param.save_folder,
                                                    no_fp16=self.train_param.no_fp16)

                    t_writer.add_scalar("SummaryLoss/overfit", validation_loss - training_loss, epoch)

        if self.train_param.warm_restart:
            print('Total number of epochs should be divisible by 30, else it will do odd things')
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 30, eta_min=1e-7)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   self.train_param.epochs + 30 if not rangered else round(
                                                                       self.train_param.epochs * 0.5))
        print("start training now!")
        if self.train_param.swa:
            # c = 15, k=3, repeat = 5
            c, k, repeat = 30, 3, self.train_param.swa_repeat
            epochs_done = self.train_param.epochs
            reboot_lr = 0
            if self.train_param.debug:
                c, k, repeat = 2, 1, 2

        for epoch in range(self.train_param.start_epoch + self.train_param.warm, self.train_param.epochs + self.train_param.warm):
            try:
                # do_epoch for one epoch
                ts = time.perf_counter()
                model.train()
                training_loss = self.step(train_loader, model, criterion, metric, self.train_param.deep_sup, optimizer, epoch, t_writer,
                                     scaler, save_folder=self.train_param.save_folder,
                                     no_fp16=self.train_param.no_fp16, patients_perf=patients_perf)
                te = time.perf_counter()
                print("Train Epoch done in " + str(te - ts) + " s")

                # Validate at the end of epoch every val step
                if (epoch + 1) % self.train_param.val == 0 and not self.train_param.full:
                    model.eval()
                    with torch.no_grad():
                        validation_loss = self.step(val_loader, model, criterion, metric, self.train_param.deep_sup, optimizer,
                                               epoch,
                                               t_writer,
                                               save_folder=self.train_param.save_folder,
                                               no_fp16=self.train_param.no_fp16, patients_perf=patients_perf)

                    t_writer.add_scalar("SummaryLoss/overfit", validation_loss - training_loss, epoch)

                    if validation_loss < best:
                        best = validation_loss
                        model_dict = model.state_dict()
                        save_checkpoint(
                            dict(
                                epoch=epoch, arch=self.train_param.arch,
                                state_dict=model_dict,
                                optimizer=optimizer.state_dict(),
                                scheduler=scheduler.state_dict(),
                            ),
                            save_folder=self.train_param.save_folder, )

                    ts = time.perf_counter()
                    print("Val epoch done in " + str(ts - te) + " s")

                if self.train_param.swa:
                    if (self.train_param.epochs - epoch - c) == 0:
                        reboot_lr = optimizer.param_groups[0]['lr']

                if not rangered:
                    scheduler.step()
                    print("scheduler stepped!")
                else:
                    if epoch / self.train_param.epochs > 0.5:
                        scheduler.step()
                        print("scheduler stepped!")

            except KeyboardInterrupt:
                print("Stopping training loop, doing benchmark")
                break

        if self.train_param.swa:
            swa_model_optim.update(model)
            print("SWA Model initialised!")
            for i in range(repeat):
                optimizer = torch.optim.Adam(model.parameters(), self.train_param.lr / 2, weight_decay=self.train_param.weight_decay)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, c + 10)
                for swa_epoch in range(c):
                    # do_epoch for one epoch
                    ts = time.perf_counter()
                    model.train()
                    swa_model.train()
                    current_epoch = epochs_done + i * c + swa_epoch
                    training_loss = self.step(train_loader, model, criterion, metric, self.train_param.deep_sup, optimizer,
                                         current_epoch, t_writer,
                                         scaler, no_fp16=self.train_param.no_fp16, patients_perf=patients_perf)
                    te = time.perf_counter()
                    print("Train Epoch done in " + str(te - ts) + " s")

                    t_writer.add_scalar("SummaryLoss/train", training_loss, current_epoch)

                    # update every k epochs and val:
                    print("cycle number: " + str(i), "swa_epoch: " + str(swa_epoch), "total_cycle_to_do " + str(repeat))
                    if (swa_epoch + 1) % k == 0:
                        swa_model_optim.update(model)
                        if not self.train_param.full:
                            model.eval()
                            swa_model.eval()
                            with torch.no_grad():
                                validation_loss = self.step(val_loader, model, criterion, metric, self.train_param.deep_sup, optimizer,
                                                       current_epoch,
                                                       t_writer, save_folder=self.train_param.save_folder, no_fp16=self.train_param.no_fp16)
                                swa_model_loss = self.step(val_loader, swa_model, criterion, metric, self.train_param.deep_sup, optimizer,
                                                      current_epoch,
                                                      t_writer, swa=True, save_folder=self.train_param.save_folder,
                                                      no_fp16=self.train_param.no_fp16)

                            t_writer.add_scalar("SummaryLoss/val", validation_loss, current_epoch)
                            t_writer.add_scalar("SummaryLoss/swa", swa_model_loss, current_epoch)
                            t_writer.add_scalar("SummaryLoss/overfit", validation_loss - training_loss, current_epoch)
                            t_writer.add_scalar("SummaryLoss/overfit_swa", swa_model_loss - training_loss, current_epoch)
                    scheduler.step()
            epochs_added = c * repeat
            save_checkpoint(
                dict(
                    epoch=self.train_param.epochs + epochs_added, arch=self.train_param.arch,
                    state_dict=swa_model.state_dict(),
                    optimizer=optimizer.state_dict()
                ),
                save_folder=self.train_param.save_folder, )
        else:
            save_checkpoint(
                dict(
                    epoch=self.train_param.epochs, arch=self.train_param.arch,
                    state_dict=model.state_dict(),
                    optimizer=optimizer.state_dict()
                ),
                save_folder=self.train_param.save_folder, )

        try:
            df_individual_perf = pd.DataFrame.from_records(patients_perf)
            print(df_individual_perf)
            df_individual_perf.to_csv(str(self.train_param.save_folder) + os.sep + "patients_indiv_perf.csv")
            reload_ckpt_bis(str(self.train_param.save_folder) + os.sep + "model_best.pth.tar", model)
            generate_segmentations(bench_loader, model, t_writer, self.train_param)
        except KeyboardInterrupt:
            print("Stopping right now!")

    def step(self, data_loader, model, criterion: EDiceLoss, metric, deep_supervision, optimizer, epoch, writer, scaler=None,
             scheduler=None, swa=False, save_folder=None, no_fp16=False, patients_perf=None):
        # Setup
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        mode = "train" if model.training else "val"
        batch_per_epoch = len(data_loader)
        progress = ProgressMeter(batch_per_epoch, [batch_time, data_time, losses], 
                                 prefix=str(mode) + "Epoch: [" + str(epoch) + "]")

        end = time.perf_counter()
        metrics = []
        print(f"fp 16: {not no_fp16}")
        data_aug = DataAugmenter(p=0.8, noise_only=False, channel_shuffling=False, drop_channnel=True).cuda()

        for i, batch in enumerate(data_loader):
            # measure data loading time
            data_time.update(time.perf_counter() - end)

            targets = batch["label"].cuda(non_blocking=True)
            inputs = batch["image"]
            nan_mask = torch.isnan(inputs)
            inputs = torch.where(nan_mask, torch.tensor(0.0, dtype=torch.float16), inputs).cuda()
            patient_id = batch["patient_id"]

            with autocast(enabled=not no_fp16):
                # data augmentation step
                if mode == "train":
                    inputs = data_aug(inputs)
                if deep_supervision:
                    segs, deeps = model(inputs)
                    if mode == "train":  # revert the data aug
                        segs, deeps = data_aug.reverse([segs, deeps])
                    loss_ = torch.stack([criterion(segs, targets)] + [criterion(deep, targets) for deep in deeps])
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

    def run_segmentation(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = self.devices
        # setup
        random.seed(self.seed)
        ngpus = torch.cuda.device_count()
        if ngpus == 0:
            raise RuntimeWarning("This will not be able to run on CPU only")

        save_folder = pathlib.Path(self.dir)
        save_folder.mkdir(parents=True, exist_ok=True)

        config_file = pathlib.Path(self.infer_param.config).resolve()
        ckpt = config_file.with_name("model_best.pth.tar")
        with config_file.open("r") as file:
            args = yaml.safe_load(file)
            args = SimpleNamespace(**args, ckpt=ckpt)
            if not hasattr(args, "normalisation"):
                args.normalisation = "minmax"

        # Create model
        model_maker = getattr(models, args.arch)
        model = model_maker(args.input_channel, 3, width=args.width, deep_supervision=args.deep_sup, 
                            norm_layer=get_norm_layer(args.norm_layer), dropout=args.dropout)

        reload_ckpt_bis(str(args.ckpt), model)

        dataset_minmax = get_datasets(self.infer_param.input_data, self.infer_param.input_patterns, self.seed, False, no_seg=True, normalisation="minmax")
        dataset_zscore = get_datasets(self.infer_param.input_data, self.infer_param.input_patterns, self.seed, False, no_seg=True, normalisation="zscore")
        loader_minmax = torch.utils.data.DataLoader(dataset_minmax, batch_size=1, num_workers=2)
        loader_zscore = torch.utils.data.DataLoader(dataset_zscore, batch_size=1, num_workers=2)

        print("Val dataset number of batch:", len(loader_minmax))
        for i, (batch_minmax, batch_zscore) in enumerate(zip(loader_minmax, loader_zscore)):
            patient_id = batch_minmax["patient_id"][0]
            ref_img_path = batch_minmax["seg_path"][0]
            crops_idx_minmax = batch_minmax["crop_indexes"]
            crops_idx_zscore = batch_zscore["crop_indexes"]
            inputs_minmax = batch_minmax["image"]
            inputs_zscore = batch_zscore["image"]
            inputs_minmax, pads_minmax = pad_batch1_to_compatible_size(inputs_minmax)
            inputs_zscore, pads_zscore = pad_batch1_to_compatible_size(inputs_zscore)
            model_preds = []
            last_norm = None
            if args.normalisation == last_norm:
                pass
            elif args.normalisation == "minmax":
                inputs = inputs_minmax.cuda()
                pads = pads_minmax
                crops_idx = crops_idx_minmax
            elif args.normalisation == "zscore":
                inputs = inputs_zscore.cuda()
                pads = pads_zscore
                crops_idx = crops_idx_zscore
            model.cuda()  # go to gpu
            with autocast():
                with torch.no_grad():
                    if self.infer_param.tta:
                        pre_segs = apply_simple_tta(model, inputs, True)
                        model_preds.append(pre_segs)
                    else:
                        if model.deep_supervision:
                            pre_segs, _ = model(inputs)
                        else:
                            pre_segs = model(inputs)

                        pre_segs = pre_segs.sigmoid_().cpu()
                    # remove pads
                    maxz, maxy, maxx = pre_segs.size(2) - pads[0], pre_segs.size(3) - pads[1], pre_segs.size(4) - pads[2]
                    pre_segs = pre_segs[:, :, 0:maxz, 0:maxy, 0:maxx].cpu()
                    print("pre_segs size", pre_segs.shape)
                    segs = torch.zeros((1, 3, 155, 240, 240))
                    segs[0, :, slice(*crops_idx[0]), slice(*crops_idx[1]), slice(*crops_idx[2])] = pre_segs[0]
                    print("segs size", segs.shape)

                    model_preds.append(segs)
                model.cpu()  # free for the next one
            pre_segs = torch.stack(model_preds).mean(dim=0)

            segs = pre_segs[0].numpy() > 0.5

            et = segs[0]
            net = np.logical_and(segs[1], np.logical_not(et))
            ed = np.logical_and(segs[2], np.logical_not(segs[1]))
            labelmap = np.zeros(segs[0].shape)
            labelmap[et] = 4
            labelmap[net] = 1
            labelmap[ed] = 2
            labelmap = sitk.GetImageFromArray(labelmap)

            ref_img = sitk.ReadImage(ref_img_path)
            labelmap.CopyInformation(ref_img)
            self.infer_param.output_path = str(self.dir) + str(patient_id) + ".nii.gz"
            print("Writing " + self.infer_param.output_path)
            sitk.WriteImage(labelmap, self.infer_param.output_path)

    def set_compartment_masks(self):
        self.mri.ede_mask = oncofem.MRI.image2mask(self.infer_param.output_path, 2)
        self.mri.act_mask = oncofem.MRI.image2mask(self.infer_param.output_path, 4)
        self.mri.nec_mask = oncofem.MRI.image2mask(self.infer_param.output_path, 1)

    def save_compartment_masks(self):
        nib.save(nib.Nifti1Image(self.mri.ede_mask, self.mri.affine), self.dir + "ede_mask.nii.gz")
        nib.save(nib.Nifti1Image(self.mri.act_mask, self.mri.affine), self.dir + "act_mask.nii.gz")
        nib.save(nib.Nifti1Image(self.mri.nec_mask, self.mri.affine), self.dir + "nec_mask.nii.gz")

def generate_segmentations(data_loader, model, writer, args):
    metrics_list = []
    for i, batch in enumerate(data_loader):
        # measure data loading time
        inputs = batch["image"]
        patient_id = batch["patient_id"][0]
        ref_path = batch["seg_path"][0]
        crops_idx = batch["crop_indexes"]
        inputs, pads = pad_batch1_to_compatible_size(inputs)
        inputs = inputs.cuda()
        with autocast():
            with torch.no_grad():
                if model.deep_supervision:
                    pre_segs, _ = model(inputs)
                else:
                    pre_segs = model(inputs)
                pre_segs = torch.sigmoid(pre_segs)
        # remove pads
        maxz, maxy, maxx = pre_segs.size(2) - pads[0], pre_segs.size(3) - pads[1], pre_segs.size(4) - pads[2]
        pre_segs = pre_segs[:, :, 0:maxz, 0:maxy, 0:maxx].cpu()
        segs = torch.zeros((1, 3, 155, 240, 240))
        segs[0, :, slice(*crops_idx[0]), slice(*crops_idx[1]), slice(*crops_idx[2])] = pre_segs[0]
        segs = segs[0].numpy() > 0.5

        et = segs[0]
        net = np.logical_and(segs[1], np.logical_not(et))
        ed = np.logical_and(segs[2], np.logical_not(segs[1]))
        labelmap = np.zeros(segs[0].shape)
        labelmap[et] = 4
        labelmap[net] = 1
        labelmap[ed] = 2
        labelmap = sitk.GetImageFromArray(labelmap)
        ref_seg_img = sitk.ReadImage(ref_path)
        ref_seg = sitk.GetArrayFromImage(ref_seg_img)
        refmap_et = ref_seg == 4
        refmap_tc = np.logical_or(refmap_et, ref_seg == 1)
        refmap_wt = np.logical_or(refmap_tc, ref_seg == 2)
        refmap = np.stack([refmap_et, refmap_tc, refmap_wt])
        patient_metric_list = calculate_metrics(segs, refmap, patient_id)
        metrics_list.append(patient_metric_list)
        labelmap.CopyInformation(ref_seg_img)
        print("Writing " + str(args.seg_folder) + str(os.sep) + str(patient_id) + ".nii.gz")
        sitk.WriteImage(labelmap, str(args.seg_folder) + str(os.sep) + str(patient_id) + ".nii.gz")
    val_metrics = [item for sublist in metrics_list for item in sublist]
    df = pd.DataFrame(val_metrics)
    overlap = df.boxplot(const.METRICS[1:], by="label", return_type="axes")
    overlap_figure = overlap[0].get_figure()
    writer.add_figure("benchmark/overlap_measures", overlap_figure)
    haussdorf_figure = df.boxplot(const.METRICS[0], by="label").get_figure()
    writer.add_figure("benchmark/distance_measure", haussdorf_figure)
    grouped_df = df.groupby("label")[const.METRICS]
    summary = grouped_df.mean().to_dict()
    for metric, label_values in summary.items():
        for label, score in label_values.items():
            writer.add_scalar("benchmark_" + str(metric) + str(os.sep) + str(label), score)
    df.to_csv(args.save_folder + os.sep + "results.csv", index=False)
