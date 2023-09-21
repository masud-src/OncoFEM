"""
Definition of process model base class. All implemented model shall be derived from this parent class.

Class:
    ProcessModel:     Base class of micro model only consist of empty interfacing functions.
"""

class ProcessModel:
    """
    The process model base class defines necessary attributes and functions for the connection to the base model.

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
