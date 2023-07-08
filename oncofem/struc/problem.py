"""
Definition of Problem Class

A problem is kept in a general description. Therefore, prototyping sub-classes for classification of parameters are
set up and can be filled problem-specific.  

Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""

from . import geometry

class Empty:
    def __init__(self):
        pass

class Parameters:
    def __init__(self):
        self.gen = Empty()
        self.time = Empty()
        self.mat = Empty()
        self.init = Empty()
        self.fem = Empty()
        self.add = Empty()
        self.ext = Empty()

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
        self.sol = Empty()
