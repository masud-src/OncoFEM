"""
# **************************************************************************#
#                                                                           #
# === Study ================================================================#
#                                                                           #
# **************************************************************************#
# Definition of study class
#
# Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
#
# --------------------------------------------------------------------------#
"""

import os.path
from .subject import Subject
import oncofem.helper.constant as constant
import pathlib

class Study:
    """
    Initializes most basic entity of optifen. Every investigation is a study. 
    A study can contain several calculations regarding different parameters, 
    geometries, models. Whole output and data should be stored in a study
    container.  
    Each study contains of input, workingdata and solution folder, herein
    the neccessary inputs can be 

        *Arguments*
            id: Study identification nummber
            dir: directory to study data
            hypothesis: Set of hypothesis (is there tumour?, if so: is it dangerous?, survival days?, what can medical agent XY do?) 
            input_data: Set of input data (.nii, .dicom, .msh, .xdmf, .csv, .jpg, .tiff)
            mech_models: Set of used mechanical models
            math_models: Set of used mathematical models
            biochem_models: Set of  used bio-chemical models
            solutions: Set of solutions
            evaluation: Set of evaluation

        *Functions*
            study = Study("name") #declares study and makes directory
    """

    def __init__(self, title):
        self.title = title
        self.dir = constant.STUDIES_DIR + title + os.sep
        self.raw_dir = self.dir + constant.RAW_DIR
        self.der_dir = self.dir + constant.DER_DIR
        self.sol_dir = self.dir + constant.SOL_DIR
        self.subjects = []

        try:
            pathlib.Path(self.dir).mkdir(parents=True, exist_ok=False)
            pathlib.Path(self.raw_dir).mkdir(parents=True, exist_ok=False)
            pathlib.Path(self.der_dir).mkdir(parents=True, exist_ok=False)
            pathlib.Path(self.sol_dir).mkdir(parents=True, exist_ok=False)
        except (FileExistsError):
            print("Study already exists")

    def create_subject(self, ident: str):
        subj = Subject(ident)
        subj.study_dir = self.dir
        self.subjects.append(subj)
        return subj
