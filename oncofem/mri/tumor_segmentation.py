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

class TumorSegmentation:
    
    def __init__(self, state: State):
        self.study_dir = state.study_dir
        self.state = state
        
    def train(self):
        pass
    
    def run_segmentation(self):
        return None
    
class Open_brats2020:

    def __init__(self):
        self.inference_params = Inference_params()
        self.training_params = Training_params()

    def inference(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = self.inference_params.devices
        inference.run_inference(self.inference_params)

    def train(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = self.training_params.devices
        train.run_train(self.training_params)

class Training_params:
    def __init__(self):
        self.input = None
        self.arch = const.OPEN_BRATS2020_TRAINING_ARCH
        self.width = const.OPEN_BRATS2020_TRAINING_WIDTH
        self.workers = const.OPEN_BRATS2020_TRAINING_WORKERS
        self.start_epoch = const.OPEN_BRATS2020_TRAINING_START_EPOCH
        self.epochs = const.OPEN_BRATS2020_TRAINING_EPOCHS
        self.batch_size = const.OPEN_BRATS2020_TRAINING_BATCH_SIZE
        self.lr = const.OPEN_BRATS2020_TRAINING_LR
        self.weight_decay = const.OPEN_BRATS2020_TRAINING_WEIGHT_DECAY
        self.resume = const.OPEN_BRATS2020_TRAINING_RESUME
        self.devices = const.OPEN_BRATS2020_DEVICES
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

class Inference_params:
    def __init__(self):
        self.config = const.OPEN_BRATS2020_DEFAULT_WEIGHTS_DIR
        self.devices = const.OPEN_BRATS2020_DEVICES
        self.on = "train"
        self.input = None  # former on
        self.tta = const.OPEN_BRATS2020_TTA
        self.seed = const.OPEN_BRATS2020_SEED
