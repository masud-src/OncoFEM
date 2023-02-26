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

import oncofem.modelling.base_model.solver
import oncofem.modelling.field_map_generator.geometry

# **************************************************************************#
#      Classes                                                              #
# **************************************************************************#
# Definition of SubClasses

class General:
    def __init__(self):
        pass

# External
class External:
    def __init__(self):
        pass

# Time-dependent Parameters
class FEM:
    def __init__(self):
        self.solver_param = oncofem.modelling.base_model.solver.SolverParam()

# Growth Parameters
class Growth:
    def __init__(self):
        pass

class Additional():
    def __init__(self):
        pass

class Initial():
    def __init__(self):
        pass

# Material Parameters and Parameters that go into weak form
class Material:
    def __init__(self):
        self.growth = Growth()

# Time-dependent Parameters
class Time:
    def __init__(self):
        pass

class Parameters:
    def __init__(self):
        self.gen = General()
        self.time = Time()
        self.mat = Material()
        self.init = Initial()
        self.fem = FEM()
        self.add = Additional()
        self.ext = External()

class Solution:
    def __init__(self):
        pass

class BioChemModels:
    def __init__(self):
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
    def __init__(self, mri=None):
        if mri is not None:
            self.mri = mri
        self.param = Parameters()
        self.geom = oncofem.modelling.field_map_generator.geometry.Geometry()
        self.sol = Solution()
        self.bmm = BioChemModels()
