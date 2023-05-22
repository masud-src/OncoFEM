"""
Definition of intern constant variables especially used for directories.
Reads information from config.ini

Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""
import os
import configparser

ONCOFEM_DIR = r"/home/marlon/Software/OncoFEM"  
CONFIG = "config.ini"
HELPER = "helper"
config = configparser.ConfigParser()
config.read(ONCOFEM_DIR + os.sep + CONFIG)

NII2MESH_DIR = config.get("directories", "NII2MESH_DIR")
CAPTK_DIR = config.get("directories", "CAPTK_DIR")
GREEDY_DIR = config.get("directories", "GREEDY_DIR")
STUDIES_DIR = config.get("directories", "STUDIES_DIR")
DER_DIR = config.get("directories", "DER_DIR")
SOL_DIR = config.get("directories", "SOL_DIR")
GENERALISATION_PATH = config.get("directories", "GENERALISATION_PATH")
TUMOR_SEGMENTATION_PATH = config.get("directories", "TUMOR_SEGMENTATION_PATH")
WHITE_MATTER_SEGMENTATION_PATH = config.get("directories", "WHITE_MATTER_SEGMENTATION_PATH")
FIELD_MAP_PATH = config.get("directories", "FIELD_MAP_PATH")

DEBUG = config.get("debug", "DEBUG")

CHANGE_HEADER = config.get("header", "CHANGE")

PATH_SRI24_T1 = ONCOFEM_DIR + os.sep + "data" + os.sep + "sri24" + os.sep + "T1.nii.gz"
PATH_SRI24_T1_BRAIN = ONCOFEM_DIR + os.sep + "data" + os.sep + "sri24" + os.sep + "T1_brain.nii.gz"
PATH_SRI24_T2 = ONCOFEM_DIR + os.sep + "data" + os.sep + "sri24" + os.sep + "T2.nii.gz"
PATH_SRI24_T2_BRAIN = ONCOFEM_DIR + os.sep + "data" + os.sep + "sri24" + os.sep + "T2_brain.nii.gz"

CWD = os.getcwd()


GENERALISATION_SHAPE = (int(config.get("generalisation", "GEN_SHAPE_X")), 
                        int(config.get("generalisation", "GEN_SHAPE_Y")), 
                        int(config.get("generalisation", "GEN_SHAPE_Z"))) 

weights_paths = config.get("open_brats2020", "DEFAULT_WEIGHTS_DIR")
OPEN_BRATS2020_DEFAULT_WEIGHTS_DIR = ONCOFEM_DIR + weights_paths
OPEN_BRATS2020_DEVICES = config.get("open_brats2020", "DEVICES")
OPEN_BRATS2020_SEED = int(config.get("open_brats2020", "SEED"))
OPEN_BRATS2020_TTA = config.getboolean("open_brats2020", "TTA")

TRAINING_OUTPUT_CHANNEL = int(config.get("open_brats2020_training", "OUTPUT_CHANNEL"))
TRAINING_RUN = config.get("open_brats2020_training", "OPEN_BRATS2020_TRAINING_RUN")
TRAINING_ARCH = config.get("open_brats2020_training", "ARCH")
TRAINING_WIDTH = int(config.get("open_brats2020_training", "WIDTH"))
TRAINING_WORKERS = int(config.get("open_brats2020_training", "WORKERS"))
TRAINING_START_EPOCH = int(config.get("open_brats2020_training", "START_EPOCH"))
TRAINING_EPOCHS = int(config.get("open_brats2020_training", "EPOCHS"))
TRAINING_BATCH_SIZE = int(config.get("open_brats2020_training", "BATCH_SIZE"))
TRAINING_LR = config.getfloat("open_brats2020_training", "LR")
TRAINING_WEIGHT_DECAY = config.getfloat("open_brats2020_training", "WEIGHT_DECAY")
TRAINING_RESUME = config.get("open_brats2020_training", "RESUME")
TRAINING_DEBUG = config.getboolean("open_brats2020_training", "DEBUG")
TRAINING_DEEP_SUP = config.getboolean("open_brats2020_training", "DEEP_SUP")
TRAINING_NO_FP16 = config.getboolean("open_brats2020_training", "NO_FP16")
TRAINING_WARM = int(config.get("open_brats2020_training", "WARM"))
TRAINING_VAL = int(config.get("open_brats2020_training", "VAL"))
TRAINING_FOLD = int(config.get("open_brats2020_training", "FOLD"))
TRAINING_NORM_LAYER = config.get("open_brats2020_training", "NORM_LAYER")
TRAINING_SWA = config.getboolean("open_brats2020_training", "SWA")
TRAINING_SWA_REPEAT = int(config.get("open_brats2020_training", "SWA_REPEAT"))
TRAINING_OPTIM = config.get("open_brats2020_training", "OPTIM")
if config.get("open_brats2020_training", "COM") == "None":
    TRAINING_COM = None
else:
    TRAINING_COM = config.get("open_brats2020_training", "COM")
TRAINING_DROPOUT = float(config.get("open_brats2020_training", "DROPOUT"))
TRAINING_WARM_RESTART = config.getboolean("open_brats2020_training", "WARM_RESTART")
TRAINING_FULL = config.getboolean("open_brats2020_training", "FULL")

HAUSSDORF = "haussdorf"
DICE = "dice"
SENS = "sens"
SPEC = "spec"
METRICS = [HAUSSDORF, DICE, SENS, SPEC]
