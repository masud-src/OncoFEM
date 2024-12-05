"""
In this sub-package of oncofem, necessary utility functions are hold.

modules:
    fem:        Definition of functions for the use of the finite element method via FEniCS are implemented.
    general:    Definition of general utils functionalities for work with the system. It is based on a linux system.
    io:         Definition of input and output interfaces and post-processing elements.
    structure:  Definition of structure elements that cluster input elements.
"""
from . import fem
from . import general
from . import io
from . import structure
