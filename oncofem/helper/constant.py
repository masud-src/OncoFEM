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
STUDIES_DIR = config.get("directories", "STUDIES_DIR")
DER_DIR = config.get("directories", "DER_DIR")
SOL_DIR = config.get("directories", "SOL_DIR")
GENERALISATION_PATH = config.get("directories", "GENERALISATION_PATH")
TUMOR_SEGMENTATION_PATH = config.get("directories", "TUMOR_SEGMENTATION_PATH")
WHITE_MATTER_SEGMENTATION_PATH = config.get("directories", "WHITE_MATTER_SEGMENTATION_PATH")
FIELD_MAP_PATH = config.get("directories", "FIELD_MAP_PATH")

DEBUG = config.get("debug", "DEBUG")

PATH_SRI24_T1 = ONCOFEM_DIR + os.sep + "data" + os.sep + "sri24" + os.sep + "T1.nii.gz"
PATH_SRI24_T1_BRAIN = ONCOFEM_DIR + os.sep + "data" + os.sep + "sri24" + os.sep + "T1_brain.nii.gz"
PATH_SRI24_T2 = ONCOFEM_DIR + os.sep + "data" + os.sep + "sri24" + os.sep + "T2.nii.gz"
PATH_SRI24_T2_BRAIN = ONCOFEM_DIR + os.sep + "data" + os.sep + "sri24" + os.sep + "T2_brain.nii.gz"

CWD = os.getcwd()


GENERALISATION_SHAPE = (int(config.get("generalisation", "GEN_SHAPE_X")), 
                        int(config.get("generalisation", "GEN_SHAPE_Y")), 
                        int(config.get("generalisation", "GEN_SHAPE_Z"))) 

weights_paths = config.get("tumor_segmentation", "DEFAULT_WEIGHTS_DIR")
TUMOR_SEGMENTATION_DEFAULT_WEIGHTS_DIR = ONCOFEM_DIR + weights_paths
TUMOR_SEGMENTATION_DEVICES = config.get("tumor_segmentation", "DEVICES")
TUMOR_SEGMENTATION_SEED = int(config.get("tumor_segmentation", "SEED"))
TUMOR_SEGMENTATION_TTA = config.getboolean("tumor_segmentation", "TTA")

TRAINING_OUTPUT_CHANNEL = int(config.get("tumor_segmentation", "OUTPUT_CHANNEL"))
TRAINING_RUN = config.get("tumor_segmentation", "TUMOR_SEGMENTATION_TRAINING_RUN")
TRAINING_ARCH = config.get("tumor_segmentation", "ARCH")
TRAINING_WIDTH = int(config.get("tumor_segmentation", "WIDTH"))
TRAINING_WORKERS = int(config.get("tumor_segmentation", "WORKERS"))
TRAINING_START_EPOCH = int(config.get("tumor_segmentation", "START_EPOCH"))
TRAINING_EPOCHS = int(config.get("tumor_segmentation", "EPOCHS"))
TRAINING_BATCH_SIZE = int(config.get("tumor_segmentation", "BATCH_SIZE"))
TRAINING_LR = config.getfloat("tumor_segmentation", "LR")
TRAINING_WEIGHT_DECAY = config.getfloat("tumor_segmentation", "WEIGHT_DECAY")
TRAINING_RESUME = config.get("tumor_segmentation", "RESUME")
TRAINING_DEBUG = config.getboolean("tumor_segmentation", "DEBUG")
TRAINING_DEEP_SUP = config.getboolean("tumor_segmentation", "DEEP_SUP")
TRAINING_NO_FP16 = config.getboolean("tumor_segmentation", "NO_FP16")
TRAINING_WARM = int(config.get("tumor_segmentation", "WARM"))
TRAINING_VAL = int(config.get("tumor_segmentation", "VAL"))
TRAINING_FOLD = int(config.get("tumor_segmentation", "FOLD"))
TRAINING_NORM_LAYER = config.get("tumor_segmentation", "NORM_LAYER")
TRAINING_SWA = config.getboolean("tumor_segmentation", "SWA")
TRAINING_SWA_REPEAT = int(config.get("tumor_segmentation", "SWA_REPEAT"))
TRAINING_OPTIM = config.get("tumor_segmentation", "OPTIM")
if config.get("tumor_segmentation", "COM") == "None":
    TRAINING_COM = None
else:
    TRAINING_COM = config.get("tumor_segmentation", "COM")
TRAINING_DROPOUT = float(config.get("tumor_segmentation", "DROPOUT"))
TRAINING_WARM_RESTART = config.getboolean("tumor_segmentation", "WARM_RESTART")
TRAINING_FULL = config.getboolean("tumor_segmentation", "FULL")
TRAINING_NULL_IMAGE = config.get("tumor_segmentation", "NULL_IMAGE")

HAUSSDORF = "haussdorf"
DICE = "dice"
SENS = "sens"
SPEC = "spec"
METRICS = [HAUSSDORF, DICE, SENS, SPEC]
