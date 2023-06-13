"""
Definition of base model class

Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""

import dolfin as df
from abc import ABC
from oncofem.struc.problem import Problem

# **************************************************************************#
#      Classes                                                              #
# **************************************************************************#
# Definition of Classes
class InitialDistribution(df.UserExpression, ABC):
    """
    t.b.d.
    """
    def __init__(self, value, **kwargs):
        self.value = value
        super().__init__(**kwargs)
    def eval_cell(self, values, x, cell):
        values[0] = self.value[cell.index]

class InitialCondition(df.UserExpression, ABC):
    """
        t.b.d.
    """
    def __init__(self, init_set,  **kwargs):
        self.init_set = self.case_distinction(init_set)
        self.size = len(init_set)
        super().__init__(**kwargs)

    def case_distinction(self, init_cond: list):
        for idx, cond in enumerate(init_cond):
            if cond is None:
                init_cond[idx] = 0.0
        return init_cond

    def eval_cell(self, values, x, cell):
        for idx,val in enumerate(values):
            if type(self.init_set[idx]) is float:
                values[idx] = self.init_set[idx]
            else:
                values[idx] = self.init_set[idx][cell.index]

    def value_shape(self):
        return self.size,

class BaseModel:
    def __init__(self):
        pass

    def set_boundaries(self, d_bound, n_bound):
        pass

    def set_initial_conditions(self, init, add):
        pass

    def set_function_spaces(self):
        pass

    def set_param(self, ip: Problem):
        pass

    def set_bio_chem_models(self, ip):
        pass

    def output(self, time):
        pass

    def set_hets_if_needed(self, field, function_space):
        pass

    def set_heterogenities(self):
        pass

    def solve(self):
        pass
