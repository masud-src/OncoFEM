"""
Definition of base model class. All implemented model shall be derived from this parent class. Since, not all models
should be continuum-mechanical models it is just an empty prototype class. 

Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""

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
        set_param:              Gives parameters to the model class. Therefore, parameters are gathered in problem class
        set_micro_models:       Sets the chosen bio-chemical model set-up on the microscale
        output:                 Defines the way the output shall be created and what shall be exported
        set_heterogenities:     Set heterogenities on the domain, if there are any, with help of set_hets_if_needed.                        
        solve:                  Method for solving the particular model
    """

    def __init__(self, *args, **kwargs):
        pass

    def set_boundaries(self, *args, **kwargs):
        pass

    def set_initial_conditions(self, *args, **kwargs):
        pass

    def set_heterogenities(self, *args, **kwargs):
        pass

    def set_param(self, *args, **kwargs):
        pass

    def set_micro_models(self, *args, **kwargs):
        pass

    def output(self, *args, **kwargs):
        pass

    def solve(self, *args, **kwargs):
        pass
