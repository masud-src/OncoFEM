"""
Definition of simple Verhulst-like kinetic with additional interplay with solid body

Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""

import dolfin as df
import ufl
from .micro_model import MicroModel


class SimpleModel(MicroModel):
    """
    Extents the simple Verhulst-like growth kinetic for mobile cancer cells resolved in a fluid constituent about the 
    swelling of the solid body. When the movable cancer cells growth beyond a threshold "cFt_min" swelling of the solid 
    body happens.
    """
    def __init__(self):
        super().__init__()
        self.prim_vars = None
        self.max_cFt = 1.
        self.max_nS = 0.8
        self.speed_cFt = 1.
        self.speed_nS = 1.
        self.cFt_min = 0.5

    def set_prim_vars(self, ansatz_functions: df.Function):
        self.prim_vars = df.split(ansatz_functions)

    def set_intern_vars(self, intern_vars: df.Function):
        self.intern_vars = df.split(intern_vars)

    def get_micro_output(self):
        u, p, nS, cFt = self.prim_vars

        H1 = df.conditional(df.gt(cFt, 0.0), 1.0, 0.0)
        H2 = df.conditional(df.gt(cFt, self.cFt_min), 1.0, 0.0)
        hat_cFt = H1 * ((self.max_cFt * 2.0)/(1.0 + ufl.exp(-self.speed_cFt * cFt)) - (self.max_cFt * 2.0) / 2.0)
        hat_nS = H2 * ((self.max_nS * 2.0)/(1.0 + ufl.exp(-self.speed_nS * nS)) - (self.max_nS * 2.0) / 2.0)

        prod_list = [None] * (len(self.prim_vars) - 2)
        prod_list[0] = hat_nS
        prod_list[1] = hat_cFt
        return prod_list
