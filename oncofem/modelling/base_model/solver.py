# **************************************************************************#
#                                                                           #
# === Solvers ==============================================================#
#                                                                           #
# **************************************************************************#
# Definition of solvers and solver parameters
#
# Co-author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
#
# --------------------------------------------------------------------------#

import dolfin

# **************************************************************************#
#      Classes                                                              #
# **************************************************************************#
# Definition of Classes

# --------------------------------------------------------------------------#
class Newton:
    """
    contains input parameters for classical Newton solver
    """

    def __init__(self):
        self.solver_type = None
        self.maxIter = None
        self.rel = None
        self.abs = None

class SolverParam:
    """
    contains all solver parameters, that can be set. Need for the sake of clean code
    """

    def __init__(self):
        self.newton = Newton()

# **************************************************************************#
#      Functions                                                            #
# **************************************************************************#
# Definition of Functions

# --------------------------------------------------------------------------#
def nonlinvarsolver(res, x, bcs, solver_param):
    """
    defines and initialises a non-linear variational problem and set up a solver scheme.

    *Arguments*
        res: residuum of problem, given by variational formulation of set of partial differential equations
        x:   solution vector, in terms of Ax=b
        bcs: dirichlet boundary conditions in form of a list
        Solver_param: Input class for solver parameters

    *Output*
        gives solver class that can be executed

    *Example*
        solver = nonlinvarsolver(residual_Momentum, x, dirichlet_boundaries, solver_parameters)
        solver.solve()
    """
    J = dolfin.derivative(res, x)
    # Initialize solver
    problem = dolfin.NonlinearVariationalProblem(res, x, bcs=bcs, J=J)
    solver = dolfin.NonlinearVariationalSolver(problem)
    #solver.parameters['nonlinear_solver'] = 'snes'
    #solver.parameters["snes_solver"]["maximum_iterations"] = 50
    #solver.parameters["snes_solver"]["report"] = False
    solver.parameters['newton_solver']['maximum_iterations'] = solver_param.newton.maxIter
    solver.parameters['newton_solver']['relative_tolerance'] = solver_param.newton.rel
    solver.parameters['newton_solver']['absolute_tolerance'] = solver_param.newton.abs
    solver.parameters['newton_solver']['linear_solver'] = solver_param.newton.solver_type
    dolfin.PETScOptions.set("-mat_mumps_cntl_1", 0.05)
    dolfin.PETScOptions.set("-mat_mumps_cntl_23", 102400)
    return solver
