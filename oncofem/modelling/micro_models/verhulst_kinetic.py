"""
Definition of simple Verhulst-like kinetic

Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""

import dolfin as df
from .micro_model import MicroModel


class VerhulstKinetic(MicroModel):
    """
    Implements a simple Verhulst-like growth kinetic for mobile cancer cells resolved in a fluid constituent. A switch
    can turn on a coupling with the solid phase. According to the amount of cancer cell concentration, the solid body
    will growth until a max level is reached. 
    
    *Attributes*:
        prim_vars:          Herein, the primary variables are hold
        flag_solid:         Bool, sets coupling from cancer cell concentration to solid body
        max_cFt:            Float, sets maximum cancer cell concentration
        max_nS:             Float, sets maximum volume fraction of solid body
        speed_cFt:          Float, controls growth speed of cancer cell concentration
        speed_nS:           Float, controls growth speed of solid body
        
    *Methods*:
        set_prim_vars:      Sets primary variables given via ansatzfunctions from a mixed element
        get_micro_output:   Calculates the Verhulst-like growth kinetic and returns the production terms
    """
    def __init__(self):
        super().__init__()
        self.prim_vars = None
        self.flag_solid = False
        self.max_cFt = 1.0
        self.max_nS = 0.8
        self.speed_cFt = 0.1
        self.speed_nS = 0.05

    def set_input(self, ansatz_functions: df.Function):
        self.prim_vars = df.split(ansatz_functions)
        
    def get_output(self):
        u, p, nS, cFt = self.prim_vars
        H1 = df.conditional(df.ge(cFt, self.max_cFt), 0.0, 1.0)
        hat_cFt = cFt * df.Constant(self.speed_cFt) * H1 * (1.0 - cFt / df.Constant(self.max_cFt))
        
        if self.flag_solid:
            hat_nS = cFt * df.Constant(self.speed_nS) * (1.0 - nS / df.Constant(self.max_nS))
        else:
            hat_nS = df.Constant(0.0)
            
        prod_list = [None] * (len(self.prim_vars) - 2)
        prod_list[0] = hat_nS
        prod_list[1] = hat_cFt
        return prod_list
