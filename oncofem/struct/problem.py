"""
# **************************************************************************#
#                                                                           #
# === Problem ==============================================================#
#                                                                           #
# **************************************************************************#
# Definition of Problem Class
#
# Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
#
# --------------------------------------------------------------------------#
"""

from optifen.solvers import SolverParam
from optifen.geom import Geometry

# **************************************************************************#
#      Classes                                                              #
# **************************************************************************#
# Definition of SubClasses

class General:
    pass

# External
class External:
    pass

# Time-dependent Parameters
class FEM:
    solver_param = SolverParam()
    pass

# Growth Parameters
class Growth:
    pass

# Material Parameters and Parameters that go into weak form
class Material:
    growth = Growth()
    pass

# Time-dependent Parameters
class Time:
    pass

class Parameters:
    gen = General()
    time = Time()
    mat = Material()
    fem = FEM()
    ext = External()
    pass

class Solution:
    pass

# --------------------------------------------------------------------------#
# --------------------------------------------------------------------------#
class Problem:
    """
    defines a Problem that describes the geometry, boundary and parameters. Should be super class for 

    *Prototypes:*
        general:
        geometry: geometrical description
        parameters: all describing parameters 
    """
    def __init__(self):
        self.param = Parameters()
        self.geom = Geometry()
        self.sol = Solution()
