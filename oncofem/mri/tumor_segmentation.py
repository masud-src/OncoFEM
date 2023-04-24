"""
# **************************************************************************#
#                                                                           #
# === Tumor segmentation ===================================================#
#                                                                           #
# **************************************************************************#
# Definition of tumor segmentation class
#
# Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
#
# --------------------------------------------------------------------------#
"""

import os
from oncofem.helper import constant as const
from oncofem.struc.state import State

import os
import pathlib
import random
from datetime import datetime
from types import SimpleNamespace

#import SimpleITK as sitk
import numpy as np
import torch
import torch.optim
import torch.utils.data
import yaml
from torch.cuda.amp import autocast

from .open_brats import models
from .open_brats.dataset import get_datasets
from .open_brats.dataset.batch_utils import pad_batch1_to_compatible_size
from .open_brats.models import get_norm_layer
from .open_brats.tta import apply_simple_tta
from .open_brats.utils import reload_ckpt_bis

class TrainParam:
    def __init__(self):
        self.arch = const.OPEN_BRATS2020_TRAINING_ARCH
        self.width = const.OPEN_BRATS2020_TRAINING_WIDTH
        self.workers = const.OPEN_BRATS2020_TRAINING_WORKERS
        self.start_epoch = const.OPEN_BRATS2020_TRAINING_START_EPOCH
        self.epochs = const.OPEN_BRATS2020_TRAINING_EPOCHS
        self.batch_size = const.OPEN_BRATS2020_TRAINING_BATCH_SIZE
        self.lr = const.OPEN_BRATS2020_TRAINING_LR
        self.weight_decay = const.OPEN_BRATS2020_TRAINING_WEIGHT_DECAY
        self.resume = const.OPEN_BRATS2020_TRAINING_RESUME
        self.debug = const.OPEN_BRATS2020_TRAINING_DEBUG
        self.deep_sup = const.OPEN_BRATS2020_TRAINING_DEEP_SUP
        self.no_fp16 = const.OPEN_BRATS2020_TRAINING_NO_FP16
        self.warm = const.OPEN_BRATS2020_TRAINING_WARM
        self.val = const.OPEN_BRATS2020_TRAINING_VAL
        self.fold = const.OPEN_BRATS2020_TRAINING_FOLD
        self.norm_layer = const.OPEN_BRATS2020_TRAINING_NORM_LAYER
        self.swa = const.OPEN_BRATS2020_TRAINING_SWA
        self.swa_repeat = const.OPEN_BRATS2020_TRAINING_SWA_REPEAT
        self.optim = const.OPEN_BRATS2020_TRAINING_OPTIM
        self.com = const.OPEN_BRATS2020_TRAINING_COM
        self.dropout = const.OPEN_BRATS2020_TRAINING_DROPOUT
        self.warm_restart = const.OPEN_BRATS2020_TRAINING_WARM_RESTART
        self.full = const.OPEN_BRATS2020_TRAINING_FULL

class InferParam:
    def __init__(self):
        self.config = const.OPEN_BRATS2020_DEFAULT_WEIGHTS_DIR
        self.output_path = None
        self.on = "train"
        self.input = None  # former on
        self.tta = const.OPEN_BRATS2020_TTA

class TumorSegmentation:

    def __init__(self, state: State):
        self.study_dir = state.study_dir
        self.state = state

        self.devices = const.OPEN_BRATS2020_DEVICES
        self.seed = const.OPEN_BRATS2020_SEED

        self.train_param = TrainParam()
        self.infer_param = InferParam()

    def run_training(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = self.devices

    def run_segmentation(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = self.devices
        # setup
        random.seed(self.seed)
        ngpus = torch.cuda.device_count()
        if ngpus == 0:
            raise RuntimeWarning("This will not be able to run on CPU only")
        print(f"Working with {ngpus} GPUs")
        print(self.infer_param.config)

        current_experiment_time = datetime.now().strftime('%Y%m%d_%T').replace(":", "")
        save_folder = pathlib.Path(f"./preds/{current_experiment_time}")
        save_folder.mkdir(parents=True, exist_ok=True)

        args_list = []
        for config in self.infer_param.config:
            config_file = pathlib.Path(config).resolve()
            print(config_file)
            ckpt = config_file.with_name("model_best.pth.tar")
            with config_file.open("r") as file:
                old_args = yaml.safe_load(file)
                old_args = SimpleNamespace(**old_args, ckpt=ckpt)
                # set default normalisation
                if not hasattr(old_args, "normalisation"):
                    old_args.normalisation = "minmax"
            print(old_args)
            args_list.append(old_args)

        if self.infer_param.on == "test":
            self.pred_folder = save_folder / f"test_segs_tta{self.infer_param.tta}"
            self.pred_folder.mkdir(exist_ok=True)
        elif self.infer_param.on == "val":
            self.pred_folder = save_folder / f"validation_segs_tta{self.infer_param.tta}"
            self.pred_folder.mkdir(exist_ok=True)
        else:
            self.pred_folder = save_folder / f"training_segs_tta{self.infer_param.tta}"
            self.pred_folder.mkdir(exist_ok=True)

        # Create model
        models_list = []
        normalisations_list = []
        for model_args in args_list:
            print(model_args.arch)
            model_maker = getattr(models, model_args.arch)

            model = model_maker(4, 3, width=model_args.width, deep_supervision=model_args.deep_sup, 
                                norm_layer=get_norm_layer(model_args.norm_layer), dropout=model_args.dropout)
            print(f"Creating {model_args.arch}")

            reload_ckpt_bis(str(model_args.ckpt), model)
            models_list.append(model)
            normalisations_list.append(model_args.normalisation)
            print("reload best weights")
            print(model)

        dataset_minmax = get_datasets(self.seed, False, no_seg=True, on=self.infer_param.on, normalisation="minmax")
        dataset_zscore = get_datasets(self.seed, False, no_seg=True, on=self.infer_param.on, normalisation="zscore")
        loader_minmax = torch.utils.data.DataLoader(dataset_minmax, batch_size=1, num_workers=2)
        loader_zscore = torch.utils.data.DataLoader(dataset_zscore, batch_size=1, num_workers=2)

        print("Val dataset number of batch:", len(loader_minmax))
        self.generate_segmentations((loader_minmax, loader_zscore), models_list, normalisations_list)

    def generate_segmentations(self, data_loaders, models, normalisations):
        for i, (batch_minmax, batch_zscore) in enumerate(zip(data_loaders[0], data_loaders[1])):
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
            for model, normalisation in zip(models, normalisations):
                if normalisation == last_norm:
                    pass
                elif normalisation == "minmax":
                    inputs = inputs_minmax.cuda()
                    pads = pads_minmax
                    crops_idx = crops_idx_minmax
                elif normalisation == "zscore":
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
            print(f"Writing {str(self.pred_folder)}/{patient_id}.nii.gz")
            sitk.WriteImage(labelmap, f"{str(self.pred_folder)}/{patient_id}.nii.gz")
