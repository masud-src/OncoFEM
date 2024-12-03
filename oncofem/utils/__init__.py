"""
In this sub-package of oncofem, necessary utils functions are based.

modules:
    constant:   Definition of intern constant variables especially used for directories. Reads from config.ini.
    fem_aux:    Definition of functions for the use of the finite element method via fenics are implemented.
    general:    Definition of general utils functionalities for work with the system. It is based on a linux system.
    io:         Definition of input and output interfaces and post-processing elements.
    structure:  Definition of medical structure elements that cluster input elements.
"""
from . import fem
from . import general
from . import io
from . import structure
