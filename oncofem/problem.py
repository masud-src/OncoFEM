"""
Definition of Problem Class. A problem is kept in a general description. Therefore, prototyping sub-classes for 
classification of parameters are set up and can be filled problem-specific. 

classes:
    Problem:        Main class of problem, that is solved via a combination of base and micro model
    Geometry:       Herein, information about the geometry are clustered
    Parameters:     Herein, parameters describing the problem are clustered
    Empty:          This is a dummy class to make a clustering of attributes possible
"""


class Problem:
    """
    defines a Problem that describes the geometry, boundary and parameters. 

    *Attributes:*
        param: holds parameter entity
        geom: holds geometry entity
    """

    def __init__(self):
        self.param = Parameters()
        self.geom = Geometry()


class Geometry:
    """
    defines the geometry of a problem

    *Attributes:*
        domain: geometrical domains
        mesh: generated mesh from xdmf format
        dim: dimension of problem
        facets: geometrical faces 
        d_bound: List of Dirichlet boundaries
        n_bound: List of Neumann boundaries
    """

    def __init__(self):
        self.domain = None
        self.mesh = None
        self.dim = None
        self.facets = None
        self.d_bound = None
        self.n_bound = None


class Parameters:
    """
    Parameters describing the problem are clustered in this class.

    *Attributes*:
        gen:        General parameters, such as titles or flags and switches
        time:       Time-dependent parameters
        mat:        Material parameters
        init:       Initial parameters
        fem:        Parameters related to finite element method (fem)
        add:        Parameters of addititives, in adaptive base models the user can add arbitrary additive components
        ext:        External paramters, such as external loads
    """

    def __init__(self):
        self.gen = Empty()
        self.time = Empty()
        self.mat = Empty()
        self.init = Empty()
        self.fem = Empty()
        self.add = Empty()
        self.ext = Empty()


class Empty:
    """
    This is a dummy class to make a clustering of attributes possible
    """

    def __init__(self):
        pass
