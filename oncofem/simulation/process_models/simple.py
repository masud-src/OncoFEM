"""
Definition of simple constant kinetic.
"""
import dolfin as df
from .process_model import ProcessModel

class Simple(ProcessModel):
    """
    Implements a simple constant growth kinetic for the solid phase.

    *Attributes*:
        prim_vars:          Herein, the primary variables are hold
        speed_nS:           Float, controls growth speed of solid body

    *Methods*:
        set_prim_vars:      Sets primary variables given via ansatzfunctions from a mixed element
        get_micro_output:   Calculates the Verhulst-like growth kinetic and returns the production terms
    """
    def __init__(self):
        super().__init__()
        self.prim_vars = None
        self.speed_nS = df.Constant(1.0e-7)

    def set_input(self, ansatz_functions: df.Function):
        self.prim_vars = df.split(ansatz_functions)

    def get_output(self):
        u, p, nS = self.prim_vars
        hat_nS = self.speed_nS

        prod_list = [None] * (len(self.prim_vars) - 2)
        prod_list[0] = hat_nS
        return prod_list
