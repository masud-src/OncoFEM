#!/usr/bin/env python
"""
# **************************************************************************#
#                                                                           #
# === Bio-chemical models ==================================================#
#                                                                           #
# **************************************************************************#
# Definition of bio-chemical model class
#
# Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
#
# --------------------------------------------------------------------------#
"""

import dolfin as df
from oncofem.struc.problem import Problem

class BioChemModel:
    """asdf"""
    def __init__(self, ip: Problem):
        """asdf"""
        self.prim_vars = None
        self.intern_vars = None
        self.cFt_ms = ip.param.mat.cFt_ms
        self.nSt_ms = ip.param.mat.nSt_ms

    def set_prim_vars(self, ansatz_functions: df.Function):
        self.prim_vars = df.split(ansatz_functions)

    def set_intern_vars(self, intern_vars: df.Function):
        self.intern_vars = df.split(intern_vars)

    def return_prod_terms(self):
        pass