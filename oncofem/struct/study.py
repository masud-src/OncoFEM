"""
# **************************************************************************#
#                                                                           #
# === General ==============================================================#
#                                                                           #
# **************************************************************************#
# Definition of general structure giving functionalities
#
# Co-author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
# Co-author: Maximilian Brodbeck <maximilian.brodbeck@isd.uni-stuttgart.de>
#
# --------------------------------------------------------------------------#
"""
import datetime
from distutils.dir_util import copy_tree
import os.path

import oncofem
from pathlib import Path

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
        self.source_dir = self.dir + constant.SOURCE_DIR
        self.raw_dir = self.dir + constant.RAW_DIR
        self.der_dir = self.dir + constant.DER_DIR
        self.sol_dir = self.dir + constant.SOL_DIR
        self.readme = str()
        self.changes = str()
        self.subjects = []
        self.hypothesis = []
        self.input_data = []
        self.mech_models = []
        self.math_models = []
        self.biochem_models = []
        self.calculations = []
        self.solutions = []
        self.evaluation = []

        try:
            Path(self.dir).mkdir(parents=True, exist_ok=False)
            Path(self.source_dir).mkdir(parents=True, exist_ok=False)
            Path(self.raw_dir).mkdir(parents=True, exist_ok=False)
            Path(self.der_dir).mkdir(parents=True, exist_ok=False)
            Path(self.sol_dir).mkdir(parents=True, exist_ok=False)

            self.gen_CHANGES()
            self.gen_README(title)

            self.save_study()
        except (FileExistsError):
            print("Study already exists")
            self.load_study(self.dir + os.sep + "save")

    def gen_CHANGES(self):
        header = constant.CHANGE_HEADER
        date = datetime.datetime.now().strftime("%m/%d/%Y")
        changes = open(self.dir + os.sep + "CHANGES.md", "w")
        self.changes = "#"+header+"\r\n\r\n\r\n"+"## 1.0.0 "+date+"\r\n"+"   - Initialized study directory"
        changes.write(self.changes)

    def gen_README(self,title):
        header = title
        readme = open(self.dir + os.sep + "README.md", "w")
        self.readme = "#"+header+"\r\n\r\n You should replace the content of this file and describe your dataset."
        readme.write(self.readme)

    def import_source_data(self, path):
        date = datetime.datetime.now().strftime("%Y_%m_%d")
        dis = self.dir+os.sep+constant.SOURCE_DIR+os.sep+os.path.split(path)[1]#+"_"+date
        copy_tree(path, dis)

    def create_subject(self, ident: str):
        subj = optifen.Subject(ident)
        subj.study_dir = self.dir
        self.subjects.append(subj)
        return subj

    def load_study(self, path):
        #study = open(path, "r")
        #lines = study.readlines()
        #self.id             = lines[0]
        #self.title          = lines[1]
        #self.dir            = lines[2]
        pass

    def save_study(self):
        #study = open(self.dir + os.sep + "save", "w+")
        #study.write(self.id+"\r\n")
        #study.write(self.title+"\r\n")
        #study.write(self.dir)
        pass    
