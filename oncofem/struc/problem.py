"""
Definition of Problem Class

Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""

import oncofem.struc.geometry

class General:
    def __init__(self):
        pass

class External:
    def __init__(self):
        pass

class FEM:
    def __init__(self):
        pass

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

class Problem:
    """
    defines a Problem that describes the geometry, boundary and parameters. 

    *Attributes:*
        mri: holds mri entity
        param: holds parameter entity
        geom: holds geometry entitiy
        base_model: holds base_model entity
        bio_model: holds bio chemical model set-up entity
        geometry: geometrical description
        parameters: all describing parameters 
    """
    def __init__(self, mri=None):
        if mri is not None:
            self.mri = mri
        self.param = Parameters()
        self.geom = oncofem.struc.geometry.Geometry()
        self.base_model = None  # oncofem.modelling.base_model.base_model.BaseModel()
        self.bio_model = None # oncofem.BioChemModel()
        self.sol = Solution()
