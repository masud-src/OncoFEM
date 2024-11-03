"""
This is the base entry point of OncoFEM. OncoFEM splits in four sub-packages that are load, when oncofem is imported.

Package:
    helper: Necessary functionalities for the infrastructure of OncoFEM. It splits into submodules for global variables
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
from .helper.constant import *
from .helper.structure import *
from . import helper
from .simulation import MRI, Problem, FieldMapGenerator, base_models, process_models
