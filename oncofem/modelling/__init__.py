"""
In this package the modelling part of OncoFEM is implemented. It splits into the three main pillars. First, the 
base model which describes the tumour with its interactions in space and time. Second, the  bio-chemical process model, 
that characterise the micro scale processes. And third, the field mapping of the input data, which makes it possible to 
preserve patient-specific input data with respect to the chosen model set-up.

packages:
    base_model: 
    micro_models:

modules:
    field_map_generator:    The field map generator interprets the given input data and creates mathematical objects 
                            with respect to the chosen model.

Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""
from .problem import Problem
from .field_map_generator import FieldMapGenerator
from . import base_models
from . import micro_models
