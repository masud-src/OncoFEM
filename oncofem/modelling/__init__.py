"""
In this package the modelling part of OncoFEM is implemented. It splits into the three main pillars. First, the 
base model which describes the tumour with its interactions in space and time. Second, the  bio-chemical process model, 
that characterise the micro scale processes. And third, the field mapping of the input data, which makes it possible to 
preserve patient-specific input data with respect to the chosen model set-up.

Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""
from .field_map_generator import FieldMapGenerator
from . import base_models
from . import micro_models