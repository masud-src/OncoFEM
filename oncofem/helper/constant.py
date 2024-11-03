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

STUDIES_DIR = config.get("directories", "STUDIES_DIR")
DER_DIR = config.get("directories", "DER_DIR")
SOL_DIR = config.get("directories", "SOL_DIR")
FIELD_MAP_PATH = config.get("directories", "FIELD_MAP_PATH")

CWD = os.getcwd()
