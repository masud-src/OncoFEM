"""
Definition of tumor segmentation class. This module represents the interface to the neural network made provided via
https://github.com/lescientifik/open_brats2020. 

classes:
    TrainParam:         Helper class to cluster the input parameters for training of neural networks
    InferParam:         Helper class to cluster the input parameters for inference of chosen neural networks. The user
                        can chose in between five different neural networks. First a full modality set and each solitary
                        modality was trained with the same hyperparameters given.
                        
    TumorSegmentation:  Interface class to control the tumor segmentation. Herein, the user can use the inference or 
                        training with particular commands.
"""
import shutil

from oncofem.helper.general import mkdir_if_not_exist
from oncofem.helper import constant as const
from . import models
from .utils import pad_batch1_to_compatible_size, determinist_collate, calculate_metrics, count_parameters, reload_ckpt_bis, reload_ckpt, WeightSWA, AverageMeter, ProgressMeter, save_metrics, save_checkpoint, get_datasets, apply_simple_tta, EDiceLoss, Ranger, Brats
from oncofem.mri.tumor_segmentation.models import get_norm_layer, DataAugmenter
import oncofem.mri
import pprint

import os
import time
import pathlib
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

class ModelParam:
    """
    
    """
    def __init__(self):
        self.arch = "EquiUnet"
        self.training_data = None
        self.full_training_data = False
        self.input_patterns = ["_t1", "_t1ce", "_t2", "_flair"]
        self.random_blank_image = False
        self.output_channel = 3
        self.width = 48
        self.optimizer = "ranger"
        self.dropout = 0.0
        self.warm_restart = False
        self.max_epochs = 200
        self.batch_size = 1
        self.lr = 0.0001
        self.weight_decay = 0.0
        self.no_float_precision_16 = False
        self.norm_layer = "group"
        self.n_warm_epochs = 3
        self.val_step_intervall = 3
        self.deep_sup = False
        self.fold = 0
        self.stochastic_weight_averaging = False
        self.stochastic_weight_averaging_repeat = 5
        self.workers = 2
        self.resume = False


class TumorSegmentation:
    """
    Tumor segmentation interface class that is used to control the training of neural networks and their interference.
    The basic code is taken from https://github.com/lescientifik/open_brats2020. In order to fit into the code of 
    OncoFEM following methods have been implemented:

    *Attributes*:
        mri:
        ts_dir:
        devices:
        dict_models:
        seed:
        save_model_folder:
        start_epoch:
        resume:
        input_data:
        weights:
        normalisation:
        output_path:
        on:
        tta:

    *Methods*:
        run_training:               runs training of a neural net with specific training parameters.
        run_segmentation:           runs segmentation or inference of a chosen neural net with a given input
        set_compartment_masks:      Sets masks for the different tumor compartments, that are identified via a specific
                                    integer. Edema is 2, Active tumor is 4 and necrotic core is 1.
        save_compartment_masks:     Save the tumor compartments into separated nifti files.

    All other functionalities come from https://github.com/lescientifik/open_brats2020 and are simply adapted to the 
    coding style of OncoFEM.
    """
    def __init__(self, mri):
        # general
        self.mri = mri
        self.ts_dir = mri.work_dir + const.TUMOR_SEGMENTATION_PATH
        self.model_param = ModelParam()

        self.devices = "0"
        self.debug = False
        self.dict_models = {"EquiUnet": models.EquiUnet}

        # training
        self.seed = 16111990
        self.save_model_folder = None
        self.start_epoch = 0

        # inference
        self.seg_file = None
        self.input_data = None
        self.weights = const.TUMOR_SEGMENTATION_WEIGHTS_DIR
        self.config = None
        self.state = mri.state
        self.normalisation = "minmax"
        self.on = "train"
        self.tta = False

    def init_inference(self) -> None:
        """
        Checks if full structural modality mode and takes best model depending on respective input parameters.
        """
        channel = [self.mri.t1_dir, self.mri.t1ce_dir, self.mri.t2_dir, self.mri.flair_dir]
        if self.mri.full_ana_modality:
            self.config = self.weights[0][1]
        elif sum(1 for var in channel if var is not None) == 1:
            index = next(index for index, value in enumerate(channel) if value is not None)
            self.config = self.weights[index + 1][1]
            self.model_param.input_patterns = [self.model_param.input_patterns[index]]
        else:
            if channel[1] is not None:
                self.config = self.weights[2][1]
            elif channel[0] is not None:
                self.config = self.weights[1][1]
            else:
                self.config = self.weights[3][1]

    def run_training(self):
        """ 
        The main training function.

        Only works for single node (be it single or multi-GPU)
        """
        scheduler = None
        val_loader = None
        c = k = None
        swa_model = None
        swa_model_optim = None
        repeat = None
        epochs_done = None

        # setup
        ngpus = torch.cuda.device_count()
        if ngpus == 0:
            raise RuntimeWarning("This will not be able to run on CPU only")

        print("Working with " + str(ngpus) + " GPUs")
        if self.model_param.optimizer.lower() == "ranger":
            self.model_param.n_warm_epochs = 0
        try:
            shutil.rmtree(self.save_model_folder + os.sep + "segs")
        except:
            print("No folder")

        config = vars(self.model_param).copy()
        mkdir_if_not_exist(self.save_model_folder)
        seg_folder = self.save_model_folder + os.sep + "segs"
        mkdir_if_not_exist(seg_folder)
        pprint.pprint(config)
        config_file = self.save_model_folder + os.sep + "hyperparam.yaml"
        with open(config_file, "w") as file:
            yaml.dump(config, file)
        t_writer = SummaryWriter(str(self.save_model_folder))

        # Create model
        print("Creating " + str(self.model_param.arch))
        self.model_param.input_channel = len(self.model_param.input_patterns)

        model_maker = self.dict_models[self.model_param.arch]

        model = model_maker(self.model_param.input_channel, self.model_param.output_channel,
            width=self.model_param.width, deep_supervision=self.model_param.deep_sup,
            norm_layer=get_norm_layer(self.model_param.norm_layer), dropout=self.model_param.dropout)

        print("total number of trainable parameters " + str(count_parameters(model)))

        if self.model_param.stochastic_weight_averaging:
            # Create the average model
            swa_model = model_maker(self.model_param.input_channel, self.model_param.output_channel,
                width=self.model_param.width, deep_supervision=self.model_param.deep_sup,
                norm_layer=get_norm_layer(self.model_param.norm_layer))
            for param in swa_model.parameters():
                param.detach_()
            swa_model = swa_model.cuda()
            swa_model_optim = WeightSWA(swa_model)

        if ngpus > 1:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda()
        print(model)
        model_file = self.save_model_folder + os.sep + "model.txt"
        with open(model_file, "w") as f:
            print(model, file=f)

        criterion = EDiceLoss().cuda()
        metric = criterion.metric
        print(metric)

        rangered = False  # needed because LR scheduling scheme is different for this optimizer
        optimizer = None
        if self.model_param.optimizer == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=self.model_param.lr,
                weight_decay=self.model_param.weight_decay, eps=1e-4)
        elif self.model_param.optimizer == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=self.model_param.lr,
                weight_decay=self.model_param.weight_decay, momentum=0.9, nesterov=True)
        elif self.model_param.optimizer == "adamw":
            print("weight decay argument will not be used. Default is 11e-2")
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.model_param.lr)
        elif self.model_param.optimizer == "ranger":
            optimizer = Ranger(model.parameters(), lr=self.model_param.lr, weight_decay=self.model_param.weight_decay)
            rangered = True

        # optionally resume from a checkpoint
        if self.model_param.resume:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, 
                lambda cur_iter: (1 + cur_iter) / (tot_iter_train * self.model_param.n_warm_epochs))
            reload_ckpt(self.model_param, model, optimizer, scheduler)

        if self.debug:
            self.model_param.epochs = 2
            self.model_param.n_warm_epochs = 0
            self.model_param.val_step_intervall = 1

        if self.model_param.full_training_data:
            train_dataset, bench_dataset = get_datasets(self.model_param.training_data, self.model_param.input_patterns,
                self.seed, self.debug, rand_blank=self.model_param.random_blank_image, full=True)

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.model_param.batch_size,
                shuffle=True, num_workers=self.model_param.workers, pin_memory=False, drop_last=True)

            bench_loader = torch.utils.data.DataLoader(bench_dataset, batch_size=1,num_workers=self.model_param.workers)
        else:
            train_dataset, val_dataset, bench_dataset = get_datasets(self.model_param.training_data, 
                self.model_param.input_patterns, self.seed, self.debug, rand_blank=self.model_param.random_blank_image, 
                fold_number=self.model_param.fold)
            
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.model_param.batch_size,
                shuffle=True, num_workers=self.model_param.workers, pin_memory=False, drop_last=True)

            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=max(1, self.model_param.batch_size // 2),
                shuffle=False, pin_memory=False, num_workers=self.model_param.workers, collate_fn=determinist_collate)

            bench_loader = torch.utils.data.DataLoader(bench_dataset, batch_size=1, num_workers=self.model_param.workers)
            print("Val dataset number of batch:", len(val_loader))

        print("Train dataset number of batch:", len(train_loader))
        # create grad scaler
        scaler = torch.cuda.amp.GradScaler()

        # Actual Train loop
        best = np.inf
        print("start warm-up now!")
        if self.model_param.n_warm_epochs != 0:
            tot_iter_train = len(train_loader)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, 
                lambda cur_iter: (1 + cur_iter)/(tot_iter_train * self.model_param.n_warm_epochs))

        patients_perf = []

        if not self.model_param.resume:
            for epoch in range(self.model_param.n_warm_epochs):
                ts = time.perf_counter()
                model.train()
                training_loss = self.step(train_loader, model, criterion, metric, self.model_param.deep_sup, optimizer,
                    epoch, t_writer, scaler, scheduler, save_folder=self.save_model_folder,
                    no_fp16=self.model_param.no_float_precision_16, patients_perf=patients_perf)
                te = time.perf_counter()
                print("Train Epoch done in " + str(te - ts) + " s")

                # Validate at the end of epoch every val step
                if (epoch + 1) % self.model_param.val_step_intervall == 0 and not self.model_param.full_training_data:
                    model.eval()
                    with torch.no_grad():
                        validation_loss = self.step(val_loader, model, criterion, metric, self.model_param.deep_sup, 
                            optimizer, epoch, t_writer, save_folder=self.save_model_folder,
                            no_fp16=self.model_param.no_float_precision_16)

                    t_writer.add_scalar("SummaryLoss/overfit", validation_loss - training_loss, epoch)

        if self.model_param.warm_restart:
            print('Total number of epochs should be divisible by 30, else it will do odd things')
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 30, eta_min=1e-7)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                self.model_param.epochs + 30 if not rangered else round(self.model_param.epochs * 0.5))
        print("start training now!")
        if self.model_param.stochastic_weight_averaging:
            # c = 15, k=3, repeat = 5
            c, k, repeat = 30, 3, self.model_param.stochastic_weight_averaging_repeat
            epochs_done = self.model_param.epochs
            if self.debug:
                c, k, repeat = 2, 1, 2

        start = self.start_epoch + self.model_param.n_warm_epochs
        end = self.model_param.epochs + self.model_param.warm_restart
        for epoch in range(start, end):
            try:
                # do_epoch for one epoch
                ts = time.perf_counter()
                model.train()
                training_loss = self.step(train_loader, model, criterion, metric, self.model_param.deep_sup, 
                    optimizer, epoch, t_writer, scaler, save_folder=self.save_model_folder,
                    no_fp16=self.model_param.no_float_precision_16, patients_perf=patients_perf)
                te = time.perf_counter()
                print("Train Epoch done in " + str(te - ts) + " s")

                # Validate at the end of epoch every val step
                if (epoch + 1) % self.model_param.val_step_intervall == 0 and not self.model_param.full_training_data:
                    model.eval()
                    with torch.no_grad():
                        validation_loss = self.step(val_loader, model, criterion, metric, self.model_param.deep_sup, 
                            optimizer, epoch, t_writer, save_folder=self.save_model_folder,
                            no_fp16=self.model_param.no_float_precision_16, patients_perf=patients_perf)

                    t_writer.add_scalar("SummaryLoss/overfit", validation_loss - training_loss, epoch)

                    if validation_loss < best:
                        best = validation_loss
                        model_dict = model.state_dict()
                        save_checkpoint(
                            dict(
                                epoch=epoch, arch=self.model_param.arch, state_dict=model_dict,
                                optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                            ),
                            save_folder=self.save_model_folder, )

                    ts = time.perf_counter()
                    print("Val epoch done in " + str(ts - te) + " s")

                if not rangered:
                    scheduler.step()
                    print("scheduler stepped!")
                else:
                    if epoch / self.model_param.epochs > 0.5:
                        scheduler.step()
                        print("scheduler stepped!")

            except KeyboardInterrupt:
                print("Stopping training loop, doing benchmark")
                break

        if self.model_param.stochastic_weight_averaging:
            swa_model_optim.update(model)
            print("SWA Model initialised!")
            for i in range(repeat):
                optimizer = torch.optim.Adam(model.parameters(), 
                    self.model_param.lr / 2, weight_decay=self.model_param.weight_decay)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, c + 10)
                for swa_epoch in range(c):
                    # do_epoch for one epoch
                    ts = time.perf_counter()
                    model.train()
                    swa_model.train()
                    current_epoch = epochs_done + i * c + swa_epoch
                    training_loss = self.step(train_loader, model, criterion, metric, self.model_param.deep_sup, 
                        optimizer, current_epoch, t_writer, scaler, no_fp16=self.model_param.no_float_precision_16, 
                        patients_perf=patients_perf)
                    te = time.perf_counter()
                    print("Train Epoch done in " + str(te - ts) + " s")

                    t_writer.add_scalar("SummaryLoss/train", training_loss, current_epoch)

                    # update every k epochs and val:
                    print("cycle number: " + str(i), "swa_epoch: " + str(swa_epoch), "total_cycle_to_do " + str(repeat))
                    if (swa_epoch + 1) % k == 0:
                        swa_model_optim.update(model)
                        if not self.model_param.full_training_data:
                            model.eval()
                            swa_model.eval()
                            with torch.no_grad():
                                validation_loss = self.step(val_loader, model, criterion, metric, 
                                    self.model_param.deep_sup, optimizer, current_epoch, t_writer, 
                                    save_folder=self.save_model_folder, no_fp16=self.model_param.no_float_precision_16)
                                swa_model_loss = self.step(val_loader, swa_model, criterion, metric, 
                                    self.model_param.deep_sup, optimizer, current_epoch, t_writer, swa=True, 
                                    save_folder=self.save_model_folder, no_fp16=self.model_param.no_float_precision_16)

                            t_writer.add_scalar("SummaryLoss/val", validation_loss, current_epoch)
                            t_writer.add_scalar("SummaryLoss/swa", swa_model_loss, current_epoch)
                            t_writer.add_scalar("SummaryLoss/overfit", validation_loss - training_loss, current_epoch)
                            t_writer.add_scalar("SummaryLoss/overfit_swa", swa_model_loss - training_loss, current_epoch)
                    scheduler.step()
            epochs_added = c * repeat
            save_checkpoint(
                dict(
                    epoch=self.model_param.epochs + epochs_added, arch=self.model_param.arch,
                    state_dict=swa_model.state_dict(), optimizer=optimizer.state_dict()
                ),
                save_folder=self.save_model_folder, )
        else:
            save_checkpoint(
                dict(
                    epoch=self.model_param.epochs, arch=self.model_param.arch,
                    state_dict=model.state_dict(), optimizer=optimizer.state_dict()
                ),
                save_folder=self.save_model_folder, )

        try:
            df_individual_perf = pd.DataFrame.from_records(patients_perf)
            print(df_individual_perf)
            df_individual_perf.to_csv(str(self.save_model_folder) + os.sep + "patients_indiv_perf.csv")
            reload_ckpt_bis(str(self.save_model_folder) + os.sep + "model_best.pth.tar", model)
            metrics_list = []
            for i, batch in enumerate(bench_loader):
                # measure data loading time
                inputs = batch["image"]
                patient_id = batch["patient_id"][0]
                ref_path = batch["seg_path"][0]
                crops_idx = batch["crop_indexes"]
                inputs, pads = pad_batch1_to_compatible_size(inputs)
                inputs = inputs.cuda()
                with torch.cuda.amp.autocast():
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
                print("Writing " + str(seg_folder) + str(os.sep) + str(patient_id) + ".nii.gz")
                sitk.WriteImage(labelmap, str(seg_folder) + str(os.sep) + str(patient_id) + ".nii.gz")
            val_metrics = [item for sublist in metrics_list for item in sublist]
            df = pd.DataFrame(val_metrics)
            overlap = df.boxplot(const.METRICS[1:], by="label", return_type="axes")
            overlap_figure = overlap[0].get_figure()
            t_writer.add_figure("benchmark/overlap_measures", overlap_figure)
            haussdorff_figure = df.boxplot(const.METRICS[0], by="label").get_figure()
            t_writer.add_figure("benchmark/distance_measure", haussdorff_figure)
            grouped_df = df.groupby("label")[const.METRICS]
            summary = grouped_df.mean().to_dict()
            for metric, label_values in summary.items():
                for label, score in label_values.items():
                    t_writer.add_scalar("benchmark_" + str(metric) + str(os.sep) + str(label), score)
            df.to_csv(self.save_model_folder + os.sep + "results.csv", index=False)
        except KeyboardInterrupt:
            print("Stopping right now!")

    def run_inference(self):
        inputs = None
        pads = None
        crops_idx = None
        os.environ['CUDA_VISIBLE_DEVICES'] = self.devices
        # setup
        ngpus = torch.cuda.device_count()
        if ngpus == 0:
            raise RuntimeWarning("This will not be able to run on CPU only")

        save_folder = pathlib.Path(self.ts_dir)
        save_folder.mkdir(parents=True, exist_ok=True)

        config_file = pathlib.Path(self.config).resolve()
        ckpt = config_file.with_name("model_best.pth.tar")
        with config_file.open("r") as file:
            args = yaml.safe_load(file)
            args = SimpleNamespace(**args, ckpt=ckpt)
            if not hasattr(args, "normalisation"):
                args.normalisation = "minmax"

        # Create model
        model_maker = self.dict_models[args.arch]
        input_channel = len(args.input_patterns)
        model = model_maker(input_channel, 3, width=args.width, deep_supervision=args.deep_sup,
            norm_layer=get_norm_layer(args.norm_layer), dropout=args.dropout)

        reload_ckpt_bis(str(args.ckpt), model)

        dataset_minmax = Brats(self.mri, self.model_param.input_patterns, False, training=False, debug=self.debug, no_seg=True, normalisation="minmax")
        dataset_zscore = Brats(self.mri, self.model_param.input_patterns, False, training=False, debug=self.debug, no_seg=True, normalisation="zscore")
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
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    if self.tta:
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
            output_segmentation = str(self.ts_dir) + str(patient_id) + ".nii.gz"
            print("Writing " + output_segmentation)
            sitk.WriteImage(labelmap, output_segmentation)
            self.mri.seg_dir = output_segmentation

    def set_compartment_masks(self):
        self.mri.ede_mask = oncofem.mri.MRI.image2mask(self.seg_file, 2)
        self.mri.act_mask = oncofem.mri.MRI.image2mask(self.seg_file, 4)
        self.mri.nec_mask = oncofem.mri.MRI.image2mask(self.seg_file, 1)

    def save_compartment_masks(self):
        nib.save(nib.Nifti1Image(self.mri.ede_mask, self.mri.affine), self.ts_dir + "ede_mask.nii.gz")
        nib.save(nib.Nifti1Image(self.mri.act_mask, self.mri.affine), self.ts_dir + "act_mask.nii.gz")
        nib.save(nib.Nifti1Image(self.mri.nec_mask, self.mri.affine), self.ts_dir + "nec_mask.nii.gz")

    def step(self, data_loader, model, criterion: EDiceLoss, metric, deep_supervision, optimizer, epoch, writer, 
             scaler=None, scheduler=None, swa=False, save_folder=None, no_fp16=False, patients_perf=None):
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

            with torch.cuda.amp.autocast(enabled=not no_fp16):
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
                    patients_perf.append(dict(id=patient_id[0], epoch=epoch, split=mode, loss=loss_.item()))

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
        with torch.cuda.amp.autocast():
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
