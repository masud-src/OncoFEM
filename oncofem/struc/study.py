"""
# **************************************************************************#
#                                                                           #
# === study module =========================================================#
#                                                                           #
# **************************************************************************#

In this module, the study class is implemented which is the main entry class
of OncoFEM, because herein all necessary directories of an investigation are
created and side products and solutions can be saved herein. 

 Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
# --------------------------------------------------------------------------#
"""

import os.path
import oncofem.struc.subject as subject
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
            title: Study identification nummber
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
        self.der_dir = self.dir + constant.DER_DIR
        self.sol_dir = self.dir + constant.SOL_DIR
        self.subjects = []

        try:
            pathlib.Path(self.dir).mkdir(parents=True, exist_ok=False)
            pathlib.Path(self.der_dir).mkdir(parents=True, exist_ok=False)
            pathlib.Path(self.sol_dir).mkdir(parents=True, exist_ok=False)
        except (FileExistsError):
            print("Study already exists")

    def create_subject(self, ident: str):
        """
        Creates a subject with a given identifier. Information about the study, including the directories are automatically given.
        Also appends the subject of the related study argument, where all subjects are gathered in a list.

        *Arguments*
        ident:      str     - Takes a string for identification

        *Return*
        subj:       subject - Returns the created subject object   
        """
        subj = subject.Subject(ident)
        subj.study_dir = self.dir
        self.subjects.append(subj)
        return subj
