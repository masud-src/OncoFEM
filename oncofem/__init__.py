"""
This is the base entry point of OncoFEM. Herein all basic functionalities are
imported.

Modules:
    constants:      First of all, necessary constants are load. Herein, also 

Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""
from .helper.constant import *
from .helper.structure import *
from . import helper
from . import interfaces
from . import mri
from . import modelling
