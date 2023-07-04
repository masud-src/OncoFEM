"""
Definition of Problem Class

A problem is kept in a general description. Therefore, prototyping sub-classes for classification of parameters are
set up and can be filled problem-specific.  

Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""

from . import geometry

class General:
    """
    General informations such as title or other comments
    """
    def __init__(self):
        pass

class External:
    """
    External quantities such as external loads
    """
    def __init__(self):
        pass

class FEM:
    """
    Parameters related to numerics
    """
    def __init__(self):
        pass

class Growth:
    """
    Particular growth parameters
    """
    def __init__(self):
        pass

class Additives:
    """
    Parameters related to additives
    """
    def __init__(self):
        pass

class Initial:
    """
    Parameters of initial state
    """
    def __init__(self):
        pass

class Material:
    """
    Material parameters
    """
    def __init__(self):
        self.growth = Growth()

class Time:
    """
    Time-dependent parameters
    """
    def __init__(self):
        pass

class Parameters:
    def __init__(self):
        self.gen = General()
        self.time = Time()
        self.mat = Material()
        self.init = Initial()
        self.fem = FEM()
        self.add = Additives()
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
        base_models: holds base_models entity
        bio_model: holds bio chemical model set-up entity
        geometry: geometrical description
        parameters: all describing parameters 
    """
    def __init__(self, mri=None):
        if mri is not None:
            self.mri = mri
        self.param = Parameters()
        self.geom = geometry.Geometry()
        self.base_model = None
        self.bio_model = None
        self.sol = Solution()
