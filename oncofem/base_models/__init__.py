"""
Base models can be defined in this package. With base models a macroscopic description of the tumor within its 
environment can be done. All models shall be imported with their constructor. See for example the two phase material.

modules:
    base_model:         Definition of base model class. All implemented model shall be derived from this parent class.
    two_phase_model:    Two-phase material. In the fluid constituent multiple components are resolved adaptively. 
"""
from .two_phase_model import TwoPhaseModel
