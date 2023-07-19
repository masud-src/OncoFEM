"""
In this sub-package of oncofem, base models can be defined. 
All models shall be imported with their constructor.

modules:
    base_model:         Definition of base model class. All implemented model shall be derived from this parent class.
    two_phase_model:    Two-phase material. In the fluid constituent multiple components are resolved adaptively. 

Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""
from .two_phase_model import TwoPhaseModel