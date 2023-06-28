"""
In this sub-package of oncofem, bio chemical model set-ups can be defined.
All models shall be imported with their constructor. 

Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""
from .verhulst_kinetic import VerhulstKinetic
from .simple_model import SimpleModel