"""
In this module the needed structure giving elements are implemented. By means of a medical application the hierarchical 
lowest layer is build by the measure. A measure basically holds the information of a single mri measurement with its 
directory, modality and possible other general information. Next layer is made up by a state, that holds several 
measurements. Most likely the measurements are related to a particular date. Thereafter is the subject layer. Herein, 
multiple states can be hold and with that a patient-specific history can be created. The most upper layer is made up by 
a study. Herein, multiple subjects are hold. Furthermore this entity serves as the general entry point for 
investigations as also a basic folder structure is created, when a study object is initialised. In this folders all 
outcomes and solutions are saved. A general way in using OncoFEM is to first initialise a study. Each upper hierarchical 
is able to create its lower level with a respective creating function.

Classes:
    measure:    A measure is the actual measure of a mri modality.
    state:      The state is the basic input into OncoFEM and represents one time step of the subject
    subject:    A subject can hold multiple states.
    study:      Base class, creates directory structure on hard disc and holds multiple subjects    

Function:
    join_path:  Concatenates different levels into one path. Is used for derivative results and solution paths of the
                entities.
"""
import os
import os.path
import pathlib
from oncofem.helper import constant

class Measure:
    """
    A measure is the actual measure of a mri modality. It usually comes raw in dicom format. In order to pre-process
    there are particular arguments for the conversion into nifti format (dir_ngz), for bias correction (dir_bia), for
    co-registration (dir_cor) and the skull stripped version (dir_sks). 

    *Attributes*:
        dir_src:          Directory of original input on hard disc
        dir_act:          Directory of actual processing step of the image
        dir_ngz:          Directory of nifti converted image
        dir_bia:          Directory of bias corrected image
        dir_cor:          Directory of co-registered image
        dir_sks:          Directory of skull stripped image
        dir_brainmask:    Directory of brain mask
        state_id:         String of state identification
        subj_id:          String of subject of measure
        study_dir:        String of study directory
        date:             Time stamp of measure
        modality:         String, identifier of modality (t1, t1gd, t2, flair, seg)
    """
    def __init__(self, path: str, modality: str):
        self.dir_src = path
        self.dir_act = path
        self.dir_ngz = None
        self.dir_bia = None
        self.dir_cor = None
        self.dir_sks = None
        self.dir_brainmask = None
        self.state_id = None
        self.subj_id = None
        self.study_dir = None
        self.date = None
        self.modality = modality

class State:
    """
    A state is an actual or reference state of a subject. A state can contain several measurements at a certain 
    time point. 

    *Attributes*:
        state_id:           String for identification of state
        subj_id:            String for identification of subject
        study_dir:          Directory of study 
        date:               Time stamp of state
        der_dir:            String of derived intermediate results directory
        sol_dir:            String of solution directory
        measures:           List of corresponding measures

    *Methods*:
        create_measure:     creates a measure that is directly bind to the state 
        set_dir:            sets the derivative and solution directories of the state
    """
    def __init__(self, ident:str):
        self.state_id = ident
        self.subj_id = None
        self.study_dir = None
        self.date = None
        self.der_dir = None
        self.sol_dir = None
        self.measures = []

    def create_measure(self, path:str, modality:str) -> Measure:
        """
        Creates a measure with a given path and modality. Initialised measures are appended in the respective list.

        *Arguments*
        path:           str     - Takes a path of the dicom or nifti files 
        modality:       str     - Takes a string identifier of the modaliry 

        *Return*
        m:              measure - Returns the created measure object   
        """
        m = Measure(path, modality)
        m.state_id = self.state_id
        m.subj_id = self.subj_id
        m.study_dir = self.study_dir
        m.date = self.date
        self.measures.append(m)
        return m

    def set_dir(self):
        """
        Sets the derivative and solution directories of the state.
        """
        self.der_dir = join_path([self.study_dir, constant.DER_DIR, self.subj_id, self.state_id])
        self.sol_dir = join_path([self.study_dir, constant.SOL_DIR, self.subj_id, self.state_id])

class Subject:
    """
    A Subject is a clinical specimen that is under investigation in some
    point. The subject usually provides patient-specific data. 

    *Attributes*:
        ident:      String for identification
        study_dir:  Directory to study
        states:     List of states

    *Methods*:
        create_state: Can create a state, that is directly bind to the subject.
    """
    def __init__(self, ident:str):
        self.subj_id = ident
        self.study_dir = None
        self.der_dir = None
        self.sol_dir = None
        self.states = []

    def create_state(self, ident:str) -> State:
        """
        Creates a state with a given identifier. Information about the study, including the directories are 
        automatically given. Also appends states, where all states are gathered in a list.

        *Arguments*
        ident:      str     - Takes a string for identification

        *Return*
        state:       state - Returns the created state object   
        """
        state = State(ident)
        state.study_dir = self.study_dir
        state.subj_id = self.subj_id
        state.set_dir()
        self.states.append(state)
        return state

    def set_dir(self):
        """
        Sets the derivative and solution directories of the state.
        """
        self.der_dir = join_path([self.study_dir, constant.DER_DIR, self.subj_id])
        self.sol_dir = join_path([self.study_dir, constant.SOL_DIR, self.subj_id])

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
    def __init__(self, title:str):
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

    def create_subject(self, ident:str) -> Subject:
        """
        Creates a subject with a given identifier. Information about the study, including the directories are automatically given.
        Also appends the subject of the related study argument, where all subjects are gathered in a list.

        *Arguments*
        ident:      str     - Takes a string for identification

        *Return*
        subj:       subject - Returns the created subject object   
        """
        subj = Subject(ident)
        subj.study_dir = self.dir
        subj.set_dir()
        self.subjects.append(subj)
        return subj

def join_path(level:list[str]):
    """
    Takes a list of folder levels and returns the concatenate path
    """
    level = [string + os.sep if not string.endswith(os.sep) else string for string in level]
    return "".join(l for l in level)
