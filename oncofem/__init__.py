"""
This is the base entry point of OncoFEM. OncoFEM splits in three sub-packages and the module 'field_map_generator' that 
are load, when oncofem is imported.

Package:
    utils:          Necessary functionalities for the infrastructure of OncoFEM. It splits into submodules for global 
                    variables general functions for the linux based system, functionalities for the finite element 
                    backend, the in and output, and medical structures to conveniently work through whole studies.

    base_models:    Herein, all implemented base models are clustered.

    process_models: Herein, all implemented process models are clustered.

Modules:
    field_map_generator:    Herein, the geometry of a problem is transformed into a mathematical mesh and all spatially 
                            distributed quantities are mapped onto that mesh
"""
from .utils import structure
from .utils.structure import Problem, ONCOFEM_DIR, STUDIES_DIR
from . import base_models, process_models
from .field_map_generator import FieldMapGenerator
