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
from . import base_models
from . import process_models
