"""
In this sub-package of oncofem, bio chemical micro model set-ups can be defined.
All models shall be imported with their constructor. 

modules:
    micro_model:        Definition of bio-chemical model base class. All implemented micro models shall be derived from 
                        this parent class.
    verhulst_kinetic:   Definition of simple Verhulst-like kinetic, with a swich to turn on a solid - concentration 
                        coupling 

Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""
from .verhulst_kinetic import VerhulstKinetic
