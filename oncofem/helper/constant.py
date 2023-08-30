"""
Definition of intern constant variables especially used for directories. Reads information from config.ini file. Only
works if global PATH Variable ONCOFEM is set.
"""
import os
import configparser

ONCOFEM_DIR = os.environ['ONCOFEM']
CONFIG = "config.ini"
config = configparser.ConfigParser()
config.read(ONCOFEM_DIR + os.sep + CONFIG)

NII2MESH_DIR = config.get("directories", "NII2MESH_DIR")
CAPTK_DIR = config.get("directories", "CAPTK_DIR")
STUDIES_DIR = config.get("directories", "STUDIES_DIR")
DER_DIR = config.get("directories", "DER_DIR")
SOL_DIR = config.get("directories", "SOL_DIR")
GENERALISATION_PATH = config.get("directories", "GENERALISATION_PATH")
TUMOR_SEGMENTATION_PATH = config.get("directories", "TUMOR_SEGMENTATION_PATH")
STRUCTURE_SEGMENTATION_PATH = config.get("directories", "STRUCTURE_SEGMENTATION_PATH")
FIELD_MAP_PATH = config.get("directories", "FIELD_MAP_PATH")

SRI24_T1 = ONCOFEM_DIR + config.get("atlas", "T1")
SRI24_T2 = ONCOFEM_DIR + config.get("atlas", "T2")

CWD = os.getcwd()

weights_paths = config.items("tumor_segmentation_weights")
TUMOR_SEGMENTATION_WEIGHTS_DIR = [(item[0], ONCOFEM_DIR + str(item[1])) for item in weights_paths]
TRAINING_RUN = config.get("tumor_segmentation_general", "TUMOR_SEGMENTATION_TRAINING_RUN")
TRAINING_NULL_IMAGE = config.get("tumor_segmentation_general", "NULL_IMAGE")

HAUSSDORFF = "haussdorff"
DICE = "dice"
SENS = "sens"
SPEC = "spec"
METRICS = [HAUSSDORFF, DICE, SENS, SPEC]
