"""
Definition of micro model base class. All implemented model shall be derived from this parent class.

Class:
    MicroModel:     Base class of micro model only consist of empty interfacing functions.
"""

class MicroModel:
    """
    The micro model base class defines necessary attributes and functions for the connection to the base model. To be 
    embedded in the OncoFEM structure, one has to create a bio-chemical model with a neccessary problem. In that way, 
    primary and secondary variables are accessible.

    *Methods:*
        set_vars:       interfacing method for input variables
        get_output:     interfacing method for output variables 
    """
    def __init__(self, *args, **kwargs):
        pass

    def set_input(self, *args, **kwargs):
        pass

    def get_output(self):
        pass
