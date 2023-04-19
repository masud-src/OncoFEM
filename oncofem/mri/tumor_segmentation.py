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

import open_brats.models
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
        self.seed = const.OPEN_BRATS2020_SEED
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
        self.on = "train"
        self.input = None  # former on
        self.tta = const.OPEN_BRATS2020_TTA
        self.seed = const.OPEN_BRATS2020_SEED

class TumorSegmentation:

    def __init__(self, state: State):
        self.study_dir = state.study_dir
        self.state = state
        
        self.devices = const.OPEN_BRATS2020_DEVICES
        
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
        print(self.config)

        current_experiment_time = datetime.now().strftime('%Y%m%d_%T').replace(":", "")
        save_folder = pathlib.Path(f"./preds/{current_experiment_time}")
        save_folder.mkdir(parents=True, exist_ok=True)

        args_list = []
        for config in self.config:
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

        if self.on == "test":
            self.pred_folder = save_folder / f"test_segs_tta{self.tta}"
            self.pred_folder.mkdir(exist_ok=True)
        elif self.on == "val":
            self.pred_folder = save_folder / f"validation_segs_tta{self.tta}"
            self.pred_folder.mkdir(exist_ok=True)
        else:
            self.pred_folder = save_folder / f"training_segs_tta{self.tta}"
            self.pred_folder.mkdir(exist_ok=True)

        # Create model
        models_list = []
        normalisations_list = []
        for model_args in args_list:
            print(model_args.arch)
            model_maker = getattr(models, model_args.arch)

            model = model_maker(4, 3, width=model_args.width, deep_supervision=model_args.deep_sup, norm_layer=get_norm_layer(model_args.norm_layer), dropout=model_args.dropout)
            print(f"Creating {model_args.arch}")

            reload_ckpt_bis(str(model_args.ckpt), model)
            models_list.append(model)
            normalisations_list.append(model_args.normalisation)
            print("reload best weights")
            print(model)

        dataset_minmax = get_datasets(self.seed, False, no_seg=True, on=self.on, normalisation="minmax")
        dataset_zscore = get_datasets(self.seed, False, no_seg=True, on=self.on, normalisation="zscore")
        loader_minmax = torch.utils.data.DataLoader(dataset_minmax, batch_size=1, num_workers=2)
        loader_zscore = torch.utils.data.DataLoader(dataset_zscore, batch_size=1, num_workers=2)

        print("Val dataset number of batch:", len(loader_minmax))
        self.generate_segmentations((loader_minmax, loader_zscore), models_list, normalisations_list)
