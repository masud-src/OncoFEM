"""
In this module, the study class is implemented which is the main entry class
of OncoFEM, because herein all necessary directories of an investigation are
created and side products and solutions can be saved herein. 

Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""

import os.path
import pathlib
from . import subject
from ..helper import constant

class Study:
    """
    Initializes most basic entity of optifen. Every investigation is a study. A study can contain several calculations 
    regarding different parameters, geometries, models. Whole output and data should be stored in a study container.  
    Each study contains of input, workingdata and solution folder, hereinthe neccessary inputs can be 

    *Attributes*:
        title: Study identification
        dir: directory to study data
        der_dir: creates a subdirectory for derived intermediate results
        sol_dir: creates a subdirectory for solutions

    *Methods*:
        create_subject: creates a subject that is directly bind to the study.
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
