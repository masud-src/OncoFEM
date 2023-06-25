"""
In this sub-package of oncofem, bio chemical model set-ups can be defined.
All models shall be imported with their constructor. 

Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""
from .gompertz_kinetic import GompertzKinetic
from .simple_model import SimpleModel