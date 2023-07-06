"""
Definition of simple Verhulst-like kinetic

Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""

import dolfin as df
from .micro_model import MicroModel


class VerhulstKinetic(MicroModel):
    """
    Implements a simple Verhulst-like growth kinetic for mobile cancer cells resolved in a fluid constituent.
    """
    def __init__(self):
        super().__init__()
        self.prim_vars = None
        self.flag_solid = False
        self.max_cFt = 1.0
        self.max_nS = 0.8
        self.speed_cFt = 0.1
        self.speed_nS = 0.05

    def set_prim_vars(self, ansatz_functions: df.Function):
        self.prim_vars = df.split(ansatz_functions)

    def set_intern_vars(self, intern_vars: df.Function):
        self.intern_vars = df.split(intern_vars)

    def get_micro_output(self):
        u, p, nS, cFt = self.prim_vars
        H1 = df.conditional(df.ge(cFt, self.max_cFt), 0.0, 1.0)
        hat_cFt = cFt * df.Constant(self.speed_cFt) * H1 * (1.0 - cFt / df.Constant(self.max_cFt))
        hat_nS = df.Constant(0.0)
        if self.flag_solid:
            hat_nS = cFt * df.Constant(self.speed_nS) * (1.0 - nS / df.Constant(self.max_nS))

        prod_list = [None] * (len(self.prim_vars) - 2)
        prod_list[0] = hat_nS
        prod_list[1] = hat_cFt
        return prod_list
