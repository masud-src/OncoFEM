"""
Definition of Geometry Class

Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""

class Geometry:
    """
    defines the geometry of a problem

    *Attributes:*
        domain: geometrical domains
        mesh: generated mesh from xdmf format
        facet_function: geometrical faces 
        d_bound: List of Dirichlet boundaries
        n_bound: List of Neumann boundaries
    """
    def __init__(self):
        self.domain = None
        self.mesh = None
        self.dim = None
        self.facet_function = None
        self.d_bound = None
        self.n_bound = None
