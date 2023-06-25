"""
In this module the needed structure giving elements are implemented. By means
of a medical application the hierarchical lowest layer is build by the
measure. A measure basically holds the information of a single mri measurement
with its directory, modality and possible other general information. Next 
layer is made up by a state, that holds several measurements. Most likely the
measurements are related to a particular date. Thereafter is the subject 
layer. Herein, multiple states can be hold and with that a patient-specific
history can be created. The most upper layer is made up by a study. Herein,
multiple subjects are hold. Furthermore this entity serves as the general
entry point for investigations as also a basic folder structure is created,
when a study object is initialised. In this folders all outcomes and solutions
are saved. A general way in using OncoFEM is to first initialise a study. Each
upper hierarchical is able to create its lower level with a respective 
creating function.

Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
# --------------------------------------------------------------------------#
"""

from .geometry import Geometry
from .measure import Measure
from .problem import Problem
from .state import State
from .study import Study
from .subject import Subject
