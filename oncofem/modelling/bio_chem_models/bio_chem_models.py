"""
Definition of bio-chemical model base class. 


Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""

import dolfin as df
from oncofem.struc.problem import Problem

class BioChemModel:
    """
    The bio-chemical model base class defines neccessary attributes and functions for the connection to the
    base model. To be embedded in the OncoFEM structure, one has to create a bio-chemical model with a neccessary
    problem. In that way, primary and secondary variables are accessible.

    *Attributes:*
        prim_vars: List, Primary variables of the enriched base model
        intern_vars: List, Chosen set of intern variables
        cFt_ms: Float, Concentration threshold of metastatic switch
        nSt_ms: Float, volume fraction threshold of metastatic switch

    *Methods:*
        set_prim_vars(ansatz_functions: dolfin.Function): takes mixed Function and splits into list, saved in prim_vars
        set_intern_vars(intern_vars: dolfin.Function): takes mixed Function and splits into list, saved in intern_vars
        return_prod_terms: template function for return of production terms 
    """
    def __init__(self):
        self.prim_vars = None
        self.intern_vars = None

    def set_prim_vars(self, ansatz_functions: df.Function):
        self.prim_vars = df.split(ansatz_functions)

    def set_intern_vars(self, intern_vars: df.Function):
        self.intern_vars = df.split(intern_vars)

    def set_param(self, ip: Problem):
        pass

    def return_prod_terms(self):
        pass
