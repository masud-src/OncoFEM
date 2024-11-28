"""
In this package the simulation part of OncoFEM is implemented. It splits into the three main pillars. First, the 
base model which describes the tumour with its interactions in space and time. Second, the  bio-chemical process model, 
that characterise the (micro scale) processes. And third, the field mapping of the input data, which makes it possible
to preserve patient-specific input data with respect to the chosen model set-up.

packages:
    base_model:             Herein, implemented base models are collected.
    process_models:         Herein, implemented process models are collected.

modules:
    field_map_generator:    The field map generator interprets the given input data and creates mathematical objects 
                            with respect to the chosen model.
    problem:                A problem is a basic entity that hold information about the problem that needs to be solved.
                            It also represents the main exchange element for passing the information.
"""

"""
This is the base entry point of OncoFEM. OncoFEM splits in four sub-packages that are load, when oncofem is imported.

Package:
    utils: Necessary functionalities for the infrastructure of OncoFEM. It splits into submodules for global variables
            general functions for the linux based system, functionalities for the finite element backend, the in and
            output, and medical structures to conveniently work through whole studies.

    interfaces: There are interfacing objects to other software. Herein, brainmage is used for skull stripping, dcm2nii
                generates 3D Nifti objects from the sliced dicom image series and nii2mesh can be used for mesh creation

    mri: This is the pre-processing module for mri image series. Herein, a module for generalisation and modules for
         segmentation of the tumor and its heterogeneous material distribution are implemented. All data is saved in a
         mri object.

    simulation: The simulation package collects base and micro models. In order to perform numerical simulations a
                problem can be set up and a field map generator is able to process gathered informations into fields
                that can be input of the used finite element backend.

Modules:
    constants: For convenience, the constant module is complete load and every global variable can be directly used.

    structure: For convenience, the structure module is complete load and every element can be directly initialised.
"""
from .utils.constant import *
from .utils import structure
from . import utils, base_models, process_models
from .problem import Problem
from .field_map_generator import FieldMapGenerator
