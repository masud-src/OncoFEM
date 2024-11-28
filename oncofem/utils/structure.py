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
"""
The handling of magenic resonance imaging series is coordinated within this mri class.

Class:
    MRI:    The base class for the pre-processing of the patient-specific input data. Main access point for all other
            mri related processes. All information is hold within this object. 
"""
from typing import Any
import nibabel as nib
import copy
import numpy as np
from os import sep
import pathlib
from oncofem.utils import constant


class Problem:
    """
    defines a Problem that describes the geometry, boundary and parameters. 

    *Attributes:*
        param: holds parameter entity
        geom: holds geometry entity
    """

    def __init__(self) -> None:
        self.param = Parameters()
        self.geom = Geometry()


class Geometry:
    """
    defines the geometry of a problem

    *Attributes:*
        domain: geometrical domains
        mesh: generated mesh from xdmf format
        dim: dimension of problem
        facets: geometrical faces 
        d_bound: List of Dirichlet boundaries
        n_bound: List of Neumann boundaries
    """

    def __init__(self):
        self.domain = None
        self.mesh = None
        self.dim = None
        self.facets = None
        self.d_bound = None
        self.n_bound = None


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
    def __init__(self, ident: str) -> None:
        self.state_id = ident
        self.subj_id = None
        self.study_dir = None
        self.date = None
        self.der_dir = None
        self.sol_dir = None
        self.measures = []

    def create_measure(self, path: str, modality: str) -> Measure:
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
    def __init__(self, ident: str):
        self.subj_id = ident
        self.study_dir = None
        self.der_dir = None
        self.sol_dir = None
        self.states = []

    def create_state(self, ident: str) -> State:
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
    def __init__(self, title: str):
        self.title = title
        self.dir = constant.STUDIES_DIR + title + sep
        self.der_dir = self.dir + constant.DER_DIR
        self.sol_dir = self.dir + constant.SOL_DIR
        self.subjects = []

        try:
            pathlib.Path(self.dir).mkdir(parents=True, exist_ok=False)
            pathlib.Path(self.der_dir).mkdir(parents=True, exist_ok=False)
            pathlib.Path(self.sol_dir).mkdir(parents=True, exist_ok=False)
        except (FileExistsError):
            print("Study already exists")

    def create_subject(self, ident: str) -> Subject:
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


class MRI:
    """
    MRI is the base class for the pre-processing of the patient-specific input data. Herein, the measures of an input
    state are sorted and the basic structural modalities are available via the respective attribute. In order to 
    homogenize and further pre-process more attributes about image properties and masks of tumor and brain tissue 
    compartments are hold. Each sub-module for a particular task is bind via its respective attribute.

    *Attributes*:
        work_dir:           String of the working directory, is set by optional state or manually
        t1_dir:             String, direction of t1 modality
        t1ce_dir:           String, direction of t1ce modality
        t2_dir:             String, direction of t2 modality
        flair_dir:          String, direction of flair modality
        seg_dir:            String, direction of segmentation
        full_ana_modality:  Bool, check if all structural modalities are given (t1, t1ce, t2, flair)
        affine:             Array of image affine, each modality is co-registered to that
        shape:              Shape of the image each modality is co-registered to
        ede_mask:           Binary mask image of the edema
        act_mask:           Binary mask image of the active core
        nec_mask:           Binary mask image of the necrotic core
        wm_mask:            Binary mask image of the white matter
        gm_mask:            Binary mask image of the gray matter
        csf_mask:           Binary mask image of the cerebro-spinal fluid
        state:              Respective input state. If initialised measures are load, full modality is checked and 
                            affine is set automatically

    *Methods*:
        set_affine:             Loads first given measurement and takes affine and shape
        load_measures:          Fills the respective arguments of the structural images and the segmentation
        isFullModality:         Checks if input state has full structural modality
        image2array:            Gives numpy array of image data
        image2mask:             Creates a mask of a given input image
        cut_area_from_image:    Cuts an area from an image
        set_state:              Sets state with working directory, loads measures, checks full modality and sets the 
                                affine.
    """
    def __init__(self, state: State = None):
        self.work_dir = None
        self.t1_dir = None
        self.t1ce_dir = None
        self.t2_dir = None
        self.flair_dir = None
        self.seg_dir = None
        self.full_ana_modality = None
        self.affine = None
        self.shape = None
        self.ede_mask = None
        self.act_mask = None
        self.nec_mask = None
        self.wm_mask = None
        self.gm_mask = None
        self.csf_mask = None
        if state is None:
            self.state = None
        else:
            self.set_state(state)

    def set_state(self, state) -> None:
        """
        Sets state with working directory, loads measures, checks full modality and sets the affine.

        :param state:
        """
        self.state = state
        self.work_dir = state.der_dir
        self.load_measures()
        self.isFullModality()
        self.set_affine()

    def set_affine(self, image: nib.Nifti1Image = None) -> None:
        """
        Sets affine and shape of first measure of included state. The optional argument takes an nibabel Nifti1Image
        and takes the first measurement of the hold state of the mri entity if no argument is given. Affine and shape
        can be accessed via self.affine and self.shape.

        *Arguments*:
            image:      Optional nib.Nifti1Image, Default is self.state.measures[0].dir_act
        """
        try:
            if image is None:
                image = nib.load(self.state.measures[0].dir_act)
            self.affine = image.affine
            self.shape = image.shape
        except:
            print("no nifti image, need to set affine after conversion.")

    def load_measures(self, state: State = None) -> None:
        """
        Loads the actual measure files and directs them to their correct modality within the mri entity. 

        *Arguments*:
            state:      Optional input state, if no argument is given, self.state.measures is taken
        """
        if state is None:
            state = self.state
        for measure in state.measures:
            if measure.modality == "t1":
                self.t1_dir = measure.dir_act
            elif measure.modality == "t1ce":
                self.t1ce_dir = measure.dir_act
            elif measure.modality == "t2":
                self.t2_dir = measure.dir_act
            elif measure.modality == "flair":
                self.flair_dir = measure.dir_act
            elif measure.modality == "seg":
                self.seg_dir = measure.dir_act

    def isFullModality(self, state: State = None) -> bool:
        """
        Checks if all structural gold standard entities are available. Returns boolean value.

        *Arguments*:
            state:      Optional input state, if no argument is given, self.state.measures is taken
        *Returns*:
            boolean value
        """
        if state is None:
            state = self.state
        list_available_modality = [measure.modality for measure in state.measures]
        list_full_modality = ["t1", "t1ce", "t2", "flair"]
        self.full_ana_modality = all(item in list_available_modality for item in list_full_modality)
        return self.full_ana_modality

    @staticmethod
    def image2array(image_dir: str) -> tuple[Any, Any, Any]:
        """
        Takes a directory of an image and gives a numpy array.

        *Arguments*:
            image_dir:      String of a Nifti image directory
        *Returns*:
            numpy array of image data, shape, affine
        """
        orig_image = nib.load(image_dir)
        return copy.deepcopy(orig_image.get_fdata()), orig_image.shape, orig_image.affine

    @staticmethod
    def image2mask(image_dir: str, compartment: int = None, inner_compartments: list[int] = None) -> np.ndarray:
        """
        Gives deep copy of original image with selected compartments.

        *Arguments*:
            image_dir:          String to Nifti image
            compartment:        Int, identifier of compartment that shall be filtered
            inner_compartments: List of inner compartments that also are included in the mask
        *Returns*:
            mask:               Numpy array of the binary mask
        """
        mask, _, _ = MRI.image2array(image_dir)
        unique = list(np.unique(mask))
        unique.remove(compartment)
        for outer in unique:
            mask[np.isclose(mask, outer)] = 0.0
        mask[np.isclose(mask, compartment)] = 1.0
        if inner_compartments is not None:
            for comp in inner_compartments:
                mask[np.isclose(mask, comp)] = 1.0
                unique.remove(comp)
        return mask


class Parameters:
    """
    Parameters describing the problem are clustered in this class.

    *Attributes*:
        gen:        General parameters, such as titles or flags and switches
        time:       Time-dependent parameters
        mat:        Material parameters
        init:       Initial parameters
        fem:        Parameters related to finite element method (fem)
        add:        Parameters of addititives, in adaptive base models the user can add arbitrary additive components
        ext:        External paramters, such as external loads
    """

    def __init__(self):
        self.gen = Empty()
        self.time = Empty()
        self.mat = Empty()
        self.init = Empty()
        self.fem = Empty()
        self.add = Empty()
        self.ext = Empty()


class Empty:
    """
    This is a dummy class to make a clustering of attributes possible
    """
    def __init__(self) -> None:
        pass


def join_path(level: list[str]) -> str:
    """
    Takes a list of folder levels and returns the concatenate path.

    :param level: 

    :return: String of concatenated path
    """
    level = [string + sep if not string.endswith(sep) else string for string in level]
    return "".join(lev for lev in level)
