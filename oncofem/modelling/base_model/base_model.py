"""
Definition of base model class. All implemented model shall be derived from this parent class. Since, not all models
should be continuum-mechanical models it is just an empty prototype class. 

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
    Defines an initial distribution of a field.

    methods:
        init:   initialises and sets the value
        eval_cell: evaluates the value at particular cells
    """
    def __init__(self, value, **kwargs):
        self.value = value
        super().__init__(**kwargs)
        
    def eval_cell(self, values, x, cell):
        values[0] = self.value[cell.index]

class InitialCondition(df.UserExpression, ABC):
    """
    Defines an initial distribution of a primary field.

    methods:
        init:   initialises and sets the initial value set
        case_distinction: sets zero if condition is None
        eval_cell: evaluates the value at particular cells
        value_shape: returns the value shape
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
    """
    The base model base class defines the layout of a base model that describes the basic entities of a tumor. 
    To be embedded in the OncoFEM structure, one has to create a base model with the included methods. In that way, 
    all other structures can work with the base model in a generalised way. It is designed in a prototype style.
    Therefore all methods are empty and the user can design its own base class.

    *Methods:*
        set_boundaries:         Sets surface boundaries, split into Dirichlet and Neumann boundaries
        set_initial_conditions: Sets initial conditions, split into init for primary variables that describe the
                                base tumor entities and an add part, where additives can be set. In that way it shall
                                be forced to implement models in a way that focus the tumor entities (healthy tissue, 
                                necrotic, active and edema part) and other ingredients, such as nutrients or VEGF
        set_function_spaces:    Sets finite element function space, don't need to be used!
        set_param:              Gives parameters to the model class. Therefore, parameters are gathered in problem class
        set_bio_chem_model:     Sets the chosen bio-chemical model set-up on the microscale
        output:                 Defines the way the output shall be created and what shall be exported
        set_hets_if_needed:     Constant fields are set constant and heterogeneous fields are set heterogeneous
        set_heterogenities:     Set heterogenities on the domain, if there are any, with help of set_hets_if_needed.                        
        solve:                  Method for solving the particular model
    """
    
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
