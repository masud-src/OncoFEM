"""
Definition of simple Gompertzian-like kinetic

Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""

import dolfin as df
import ufl
from .bio_chem_models import BioChemModel


class GompertzKinetic(BioChemModel):
    """
    Implements a simple Gompertzian-like growth kinetic for mobile cancer cells resolved in a fluid constituent.
    """
    def __init__(self):
        super().__init__()
        self.max_cFt = 1.2
        self.speed = 0.01

    def return_prod_terms(self):
        u, p, nS, cFt = self.prim_vars

        H1 = df.conditional(df.gt(cFt, 0.0), cFt/self.max_cFt, 0.0)
        help1 = 2*self.max_cFt*self.speed*ufl.exp(-self.speed*cFt)
        help2 = (ufl.exp(-self.speed*cFt) + 1) * (ufl.exp(-self.speed*cFt) + 1) 
        hat_cFt = df.conditional(df.gt(cFt, self.max_cFt), 0.0, H1 * (help1/help2))
        hat_nS = df.Constant(0.0)

        prod_list = [None] * (len(self.prim_vars) - 2)
        prod_list[0] = hat_nS
        prod_list[1] = hat_cFt
        return prod_list
