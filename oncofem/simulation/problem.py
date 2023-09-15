"""
Definition of Problem Class. A problem is kept in a general description. Therefore, prototyping sub-classes for 
classification of parameters are set up and can be filled problem-specific. 

classes:
    Problem:        Main class of problem, that is solved via a combination of base and micro model
    Empty:          This is a dummy class to make a clustering of attributes possible
    Geometry:       Herein, informations about the geometry are clustered
    Parameters:     Herein, parameters describing the problem are clustered
"""
class Problem:
    """
    defines a Problem that describes the geometry, boundary and parameters. 

    *Attributes:*
        mri: holds mri entity
        param: holds parameter entity
        geom: holds geometry entitiy
        base_model: holds base_model entity
        micro_model: holds micro model set-up entity
        sol: holds solution of particular model set up
    """
    def __init__(self, mri=None):
        if mri is not None:
            self.mri = mri
        self.param = Parameters()
        self.geom = Geometry()
        self.base_model = None
        self.micro_model = None
        self.sol = Empty()

class Empty:
    """
    This is a dummy class to make a clustering of attributes possible
    """
    def __init__(self):
        pass

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
