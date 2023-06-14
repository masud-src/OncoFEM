"""
Definition of diffusive base model

Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""
import oncofem as of
import oncofem.helper.solver as solv
import oncofem.helper.general as gen
from oncofem.helper.io import write_field2xdmf
import dolfin as df
import ufl
from oncofem.modelling.base_model.base_model import BaseModel
from oncofem.modelling.base_model.base_model import InitialDistribution, InitialCondition


class Diffusion(BaseModel):
    """
    t.b.d.
    """
    def __init__(self):
        super().__init__()
        # general info
        self.output_file = None

        # FEM parameters
        self.tensor_order = 0
        self.ele_type = "CG"
        self.ele_order = 1
        self.finite_element = None
        self.function_space = None
        self.ansatz_function = None
        self.test_function = None
        self.DG0_sca = None
        self.CG1_sca = None
        self.sol = None
        self.sol_old = None

        # geometry parameters
        self.mesh = None
        self.dim = None
        self.n_bound = None
        self.d_bound = None
        self.dx = None

        # weak form, output and solver
        self.f = None
        self.residuum = None
        self.solver = None

        # spatial varying material parameters
        self.a = None

        # initial conditions
        self.u_0 = None

        # time parameters
        self.time = None
        self.T_end = None
        self.output_interval = None
        self.dt = None

        # solver paramters
        self.solver_type = "lu"
        self.maxIter = 20
        self.rel = 1E-7
        self.abs = 1E-8

    def set_boundaries(self, d_bound, n_bound):
        self.d_bound = d_bound
        self.n_bound = n_bound

    def assign_if_function(self, var, index):
        if type(var) is df.Function:
            df.assign(self.sol.sub(index), var)
            df.assign(self.sol_old.sub(index), var)

    def actualize_prod_terms(self):
        if self.f is not None:
            self.f.assign(df.project(self.f, self.CG1_sca))
        else:
            self.f = df.Constant(0.0)

    def set_initial_conditions(self, init):
        """
        Sets initial condition for adaptive system. Can take scalars, distribution from MeshFunctions and Functions.
        """
        # set initial var
        self.u_0 = init.u_0

        # collect for interpolation
        init_set = self.set_hets_if_needed(self.u_0, self.function_space)
        self.sol.interpolate(init_set)
        self.sol_old.interpolate(init_set)

        self.assign_if_function(self.u_0, 0)

        # production terms
        self.actualize_prod_terms()

    def set_function_spaces(self):
        """
        sets function space for primary variable u and for internal variables
        """
        self.finite_element = df.FiniteElement(self.ele_type, self.mesh.ufl_cell(), self.ele_order)
        self.function_space = df.FunctionSpace(self.mesh, self.finite_element)
        self.CG1_sca = df.FunctionSpace(self.mesh, "CG", 1)
        self.DG0_sca = df.FunctionSpace(self.mesh, "DG", 0)
        self.ansatz_function = df.Function(self.function_space)
        self.test_function = df.TestFunction(self.function_space)

    def set_param(self, input):
        self.output_file = input.param.gen.output_file
        self.mesh = input.geom.mesh
        self.dx = df.Measure("dx")
        self.n_bound = input.geom.n_bound
        self.d_bound = input.geom.d_bound
        self.a = input.param.mat.a
        self.solver_type = "lu"
        self.maxIter = 20
        self.rel = 1E-7
        self.abs = 1E-8
        self.dt = input.param.time.dt
        self.T_end = input.param.time.T_end

    def set_bio_chem_models(self, prod_term):
        self.f = prod_term

    def output(self, time) -> None:
        write_field2xdmf(self.output_file, self.sol, "u", time)

    def set_hets_if_needed(self, field, function_space):
        if type(field) is float:
            field = df.Constant(field)
        else:
            new_field = df.Function(function_space)
            new_field.interpolate(field)
        return field

    def set_heterogenities(self):
        self.a = self.set_hets_if_needed(self.a, function_space=self.DG0_sca)

    def set_weak_form(self) -> None:
        ##############################################################################
        # Get Ansatz and test functions
        #######################################
        self.sol_old = df.Function(self.function_space)  # old primaries
        u = self.ansatz_function
        _u = self.test_function
        u_n = self.sol_old

        # residual
        res = u * _u * self.dx + self.dt * ufl.dot(ufl.grad(u), ufl.grad(_u)) * self.dx - (u_n + self.dt * self.f) * _u * self.dx
        if self.n_bound is not None:
            res += self.n_bound
        ##############################################################################
        self.residuum = res

    def set_solver(self):
        # Make sure quadrature_degree stays at 2
        prm = df.parameters["form_compiler"]
        prm["quadrature_degree"] = 2
        self.sol = self.ansatz_function
        solver = solv.Solver()
        solver.solver_type = self.solver_type
        solver.abs = self.abs
        solver.rel = self.rel
        solver.maxIter = self.maxIter
        self.solver = solver.set_non_lin_solver(self.residuum, self.sol, self.d_bound)

    def solve(self):
        # Initialize  and time loop
        t = 0.0
        out_count = 0.0
        self.output(t)
        print("Initial step is written")
        while t < self.T_end:
            # Increment solution time
            t = t + self.dt
            out_count += self.dt
            # Calculate current solution
            n_iter, converged = self.solver.solve()
            # actualize prod terms
            self.actualize_prod_terms()
            # Output solution
            if out_count >= self.output_interval:  #-self.dt:
                print("Time: {}".format(t), "  ", "Converged in steps: {}".format(n_iter))
                out_count = 0.0
                self.output(t)
            # Update history fields
            self.sol_old.assign(self.sol)
