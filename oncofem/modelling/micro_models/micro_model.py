"""
Definition of bio-chemical model base class. 


Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""

class MicroModel:
    """
    The bio-chemical model base class defines neccessary attributes and functions for the connection to the
    base model. To be embedded in the OncoFEM structure, one has to create a bio-chemical model with a neccessary
    problem. In that way, primary and secondary variables are accessible.

    *Methods:*
        set_prim_vars:      takes mixed Function and splits into list, saved in prim_vars
        set_intern_vars:    takes mixed Function and splits into list, saved in intern_vars
        get_micro_output:   template function for return of production terms 
    """
    def __init__(self, *args, **kwargs):
        pass

    def set_prim_vars(self, *args, **kwargs):
        pass

    def set_intern_vars(self, *args, **kwargs):
        pass

    def set_param(self, *args, **kwargs):
        pass

    def get_micro_output(self):
        pass
