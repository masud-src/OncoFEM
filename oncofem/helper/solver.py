"""
Definition of solver for non-linear finite-element calculations

Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""

import dolfin as df

class Solver:
    """
    contains all solver parameters, that can be set. Need for the sake of clean code
    """
    def __init__(self):
        self.solver_type = "mumps"
        self.maxIter = 20
        self.rel = 1.e-7
        self.abs = 1.e-6
        self.mumps_cntl_1 = 0.05
        self.mumps_icntl_23 = 102400

    def set_non_lin_solver(self, res, x, bcs):
        """
        defines and initialises a non-linear variational problem and set up a solver scheme.

        *Arguments*
            res: residuum of problem, given by variational formulation of set of partial differential equations
            x:   solution vector, in terms of Ax=b
            bcs: dirichlet boundary conditions in form of a list

        *Output*
            gives solver class that can be executed

        *Example*
            solver = solver_object.set_non_lin_solver(residual_Momentum, x, dirichlet_boundaries)
            solver.solve()
        """
        J = df.derivative(res, x)
        problem = df.NonlinearVariationalProblem(res, x, bcs=bcs, J=J)
        solver = df.NonlinearVariationalSolver(problem)
        solver.parameters['newton_solver']['maximum_iterations'] = self.maxIter
        solver.parameters['newton_solver']['relative_tolerance'] = self.rel
        solver.parameters['newton_solver']['absolute_tolerance'] = self.abs
        solver.parameters['newton_solver']['linear_solver'] = self.solver_type
        if self.solver_type == "mumps":
            df.PETScOptions.set("-mat_mumps_cntl_1", self.mumps_cntl_1)  # relative pivoting threshold
            df.PETScOptions.set("-mat_mumps_icntl_23", self.mumps_icntl_23)  # max size of the working memory (MB) that can allocate per processor
        return solver
