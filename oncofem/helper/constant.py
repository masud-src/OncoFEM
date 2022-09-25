"""
# **************************************************************************#
#                                                                           #
# === Constant =============================================================#
#                                                                           #
# **************************************************************************#
# Definition of intern constant variables especially used for directories
#
# Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
#
# --------------------------------------------------------------------------#
"""

import os
import configparser

ONCOFEM_DIR = r"/home/marlon/Software/OncoFEM/oncofem"  
#ONCOFEM_DIR = r"optifen/"  # relative path!
CONFIG = "config.ini"
HELPER = "helper"
config = configparser.ConfigParser()
config.read(ONCOFEM_DIR + os.sep + HELPER + os.sep + CONFIG)

STUDIES_DIR = config.get("directories", "STUDIES_DIR")
SOURCE_DIR = config.get("directories", "SOURCE_DIR")
NII2MESH_DIR = config.get("directories", "NII2MESH_DIR")
GREEDY_DIR = config.get("directories", "GREEDY_DIR")
RAW_DIR = config.get("directories", "RAW_DIR")
DER_DIR = config.get("directories", "DER_DIR")
SOL_DIR = config.get("directories", "SOL_DIR")
GENERALISATION_PATH = config.get("directories", "GENERALISATION_PATH")

CHANGE_HEADER = config.get("header", "CHANGE")

PATH_SRI24_T1 = ONCOFEM_DIR + "data" + os.sep + "NITRC" + os.sep + "sri24_spm8" + os.sep + "templates" + os.sep + "T1.nii"
PATH_SRI24_T1_BRAIN = ONCOFEM_DIR + "data" + os.sep + "NITRC" + os.sep + "sri24_spm8" + os.sep + "templates" + os.sep + "T1_brain.nii"
PATH_SRI24_T2 = ONCOFEM_DIR + "data" + os.sep + "NITRC" + os.sep + "sri24_spm8" + os.sep + "templates" + os.sep + "T2.nii"
PATH_SRI24_T2_BRAIN = ONCOFEM_DIR + "data" + os.sep + "NITRC" + os.sep + "sri24_spm8" + os.sep + "templates" + os.sep + "T2_brain.nii"

CWD = os.getcwd()


GENERALISATION_SHAPE = (int(config.get("gen", "GEN_SHAPE_X")), int(config.get("gen", "GEN_SHAPE_Y")), int(config.get("gen", "GEN_SHAPE_X"))) 

weights_paths = config.get("open_brats2020", "DEFAULT_WEIGHTS_DIR").split()
OPEN_BRATS2020_DEFAULT_WEIGHTS_DIR = [ONCOFEM_DIR + path for path in weights_paths]
OPEN_BRATS2020_DEVICES = config.get("open_brats2020", "DEVICES")
OPEN_BRATS2020_SEED = int(config.get("open_brats2020", "SEED"))
OPEN_BRATS2020_TTA = config.getboolean("open_brats2020", "TTA")
OPEN_BRATS2020_TRAIN_FOLDERS = config.get("open_brats2020", "OPEN_BRATS2020_TRAIN_FOLDERS")
OPEN_BRATS2020_VAL_FOLDER = config.get("open_brats2020", "OPEN_BRATS2020_VAL_FOLDER")
OPEN_BRATS2020_TEST_FOLDER = config.get("open_brats2020", "OPEN_BRATS2020_TEST_FOLDER")

OPEN_BRATS2020_TRAINING_RUN = config.get("open_brats2020_training", "OPEN_BRATS2020_TRAINING_RUN")
OPEN_BRATS2020_TRAINING_ARCH = config.get("open_brats2020_training", "ARCH")
OPEN_BRATS2020_TRAINING_WIDTH = int(config.get("open_brats2020_training", "WIDTH"))
OPEN_BRATS2020_TRAINING_WORKERS = int(config.get("open_brats2020_training", "WORKERS"))
OPEN_BRATS2020_TRAINING_START_EPOCH = int(config.get("open_brats2020_training", "START_EPOCH"))
OPEN_BRATS2020_TRAINING_EPOCHS = int(config.get("open_brats2020_training", "EPOCHS"))
OPEN_BRATS2020_TRAINING_BATCH_SIZE = int(config.get("open_brats2020_training", "BATCH_SIZE"))
OPEN_BRATS2020_TRAINING_LR = config.getfloat("open_brats2020_training", "LR")
OPEN_BRATS2020_TRAINING_WEIGHT_DECAY = config.getfloat("open_brats2020_training", "WEIGHT_DECAY")
OPEN_BRATS2020_TRAINING_RESUME = config.get("open_brats2020_training", "RESUME")
OPEN_BRATS2020_TRAINING_DEBUG = config.getboolean("open_brats2020_training", "DEBUG")
OPEN_BRATS2020_TRAINING_DEEP_SUP = config.getboolean("open_brats2020_training", "DEEP_SUP")
OPEN_BRATS2020_TRAINING_NO_FP16 = config.getboolean("open_brats2020_training", "NO_FP16")
OPEN_BRATS2020_TRAINING_WARM = int(config.get("open_brats2020_training", "WARM"))
OPEN_BRATS2020_TRAINING_VAL = int(config.get("open_brats2020_training", "VAL"))
OPEN_BRATS2020_TRAINING_FOLD = int(config.get("open_brats2020_training", "FOLD"))
OPEN_BRATS2020_TRAINING_NORM_LAYER = config.get("open_brats2020_training", "NORM_LAYER")
OPEN_BRATS2020_TRAINING_SWA = config.getboolean("open_brats2020_training", "SWA")
OPEN_BRATS2020_TRAINING_SWA_REPEAT = int(config.get("open_brats2020_training", "SWA_REPEAT"))
OPEN_BRATS2020_TRAINING_OPTIM = config.get("open_brats2020_training", "OPTIM")
if config.get("open_brats2020_training", "COM") == "None":
    OPEN_BRATS2020_TRAINING_COM = None
else:
    OPEN_BRATS2020_TRAINING_COM = config.get("open_brats2020_training", "COM")
OPEN_BRATS2020_TRAINING_DROPOUT = float(config.get("open_brats2020_training", "DROPOUT"))
OPEN_BRATS2020_TRAINING_WARM_RESTART = config.getboolean("open_brats2020_training", "WARM_RESTART")
OPEN_BRATS2020_TRAINING_FULL = config.getboolean("open_brats2020_training", "FULL")
