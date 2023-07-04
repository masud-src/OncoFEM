"""
Definition of two-phase material. In the fluid constituent multiple 
components can be resolved adaptively.

Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""
import time
import oncofem.helper.auxillaries as aux
from oncofem.helper.auxillaries import InitialCondition
import oncofem.helper.general as gen
from oncofem.struc.problem import Problem
from oncofem.helper.io import write_field2xdmf
import dolfin as df
import ufl
from oncofem.modelling.base_models.base_model import BaseModel

class TwoPhaseModel(BaseModel):

    def __init__(self):
        super().__init__()
        # general info
        self.output_file = None
        self.flag_defSplit = False

        # FEM paramereters
        self.solver_param = None
        self.prim_vars_list = ["u", "p", "nS"]
        self.n_init_prim_vars = len(self.prim_vars_list)
        self.tensor_order = [1, 0, 0]
        self.ele_types = ["CG", "CG", "CG"]
        self.ele_orders = [2, 1, 1]
        self.finite_element = None
        self.function_space = None
        self.ansatz_functions = None
        self.test_functions = None
        self.DG0 = None
        self.CG1_sca = None
        self.CG1_vec = None
        self.CG1_ten = None
        self.sol = None
        self.sol_old = None

        # geometry paramereters
        self.mesh = None
        self.dim = None
        self.n_bound = None
        self.d_bound = None

        # weak form, output and solver
        self.hatnS = None
        self.hatrhoFkappa = []
        self.bio_terms = []
        self.residuum = None
        self.intern_output = None
        self.solver = None

        # intrinsic material parameters
        self.rhoSR = None
        self.rhoFR = None
        self.gammaFR = None
        self.R = None
        self.Theta = None

        # spatial varying material parameters
        self.kF = None
        self.lambdaS = None
        self.muS = None
        self.DFt = None

        # initial conditions
        self.uS_0S = None
        self.p_0S = None
        self.nS_0S = None

        # additional concentrations
        self.cFkappa = None
        self.molFkappa = None
        self.DFkappa = None
        self.cFkappa_0S = None

        # time parameters
        self.time = None
        self.T_end = None
        self.output_interval = None
        self.dt = None

    def set_boundaries(self, d_bound, n_bound):
        self.d_bound = d_bound
        self.n_bound = n_bound    

    def assign_if_function(self, var, index):
        if type(var) is df.Function:
            df.assign(self.sol.sub(index), var)
            df.assign(self.sol_old.sub(index), var)

    def actualize_prod_terms(self):
        if self.prod_terms[0] is not None:
            self.hatnS.assign(df.project(self.prod_terms[0], self.CG1_sca))
        else:
            self.hatnS = df.Constant(0.0)

        for idx in range(1, len(self.prod_terms)):
            if self.prod_terms[idx] is not None:
                self.hatrhoFkappa[idx-1].assign(df.project(self.prod_terms[idx], self.CG1_sca))
            else:
                self.hatrhoFkappa[idx-1] = df.Constant(0.0)

    def set_initial_conditions(self, init, add):
        """
        Sets initial condition for adaptive system. Can take scalars, distribution from MeshFunctions and Functions.
        """
        # set intern vars
        self.uS_0S = init.uS_0S
        self.p_0S = init.p_0S
        self.nS_0S = init.nS_0S
        if hasattr(add, "prim_vars"):
            self.cFkappa_0S = add.cFkappa_0S
        # collect for interpolation
        init_set = []
        if type(init.uS_0S) is df.Function:
            init_set = [None] * self.dim
        if type(init.uS_0S) is list:
            for i in range(self.dim):
                init_set.append(gen.check_if_type(init.uS_0S[i], df.Function, None))
        init_set.append(gen.check_if_type(init.p_0S, df.Function, None))
        init_set.append(gen.check_if_type(init.nS_0S, df.Function, None))
        if self.cFkappa_0S is not None:
            for cFd_0S in self.cFkappa_0S:
                init_set.append(gen.check_if_type(cFd_0S, df.Function, None))
        self.sol.interpolate(InitialCondition(init_set))
        self.sol_old.interpolate(InitialCondition(init_set))

        self.assign_if_function(self.uS_0S, 0)
        self.assign_if_function(self.p_0S, 1)
        self.assign_if_function(self.nS_0S, 2)
        if self.cFkappa_0S is not None:
            for idx, cFd_0S in enumerate(self.cFkappa_0S):
                self.assign_if_function(cFd_0S, 3+idx)
        # production terms
        self.actualize_prod_terms()

    def set_function_spaces(self):
        """
            sets function space for primary variables u, p, nS, cFdelta and for internal variables
        """
        elements = []
        for idx, e_type in enumerate(self.ele_types):
            if self.tensor_order[idx] == 0:
                elements.append(df.FiniteElement(e_type, self.mesh.ufl_cell(), self.ele_orders[idx]))
            elif self.tensor_order[idx] == 1:
                elements.append(df.VectorElement(e_type, self.mesh.ufl_cell(), self.ele_orders[idx]))
            elif self.tensor_order[idx] == 2:
                elements.append(df.TensorElement(e_type, self.mesh.ufl_cell(), self.ele_orders[idx]))
        self.finite_element = ufl.MixedElement(elements)
        self.function_space = df.FunctionSpace(self.mesh, self.finite_element)
        self.DG0 = df.FunctionSpace(self.mesh, "DG", 0)
        self.CG1_sca = df.FunctionSpace(self.mesh, "CG", 1)
        self.CG1_vec = df.VectorFunctionSpace(self.mesh, "CG", 1)
        self.CG1_ten = df.TensorFunctionSpace(self.mesh, "CG", 1)
        self.ansatz_functions = df.Function(self.function_space)
        self.test_functions = df.TestFunction(self.function_space)

    def set_param(self, ip: Problem):
        """
        sets parameter needed for model class
        """
        # general parameters
        self.output_file = ip.param.gen.output_file
        self.flag_defSplit = ip.param.gen.flag_defSplit

        # time parameters
        self.T_end = ip.param.time.T_end
        self.output_interval = ip.param.time.output_interval
        self.dt = ip.param.time.dt

        # material parameters base model
        self.rhoSR = df.Constant(ip.param.mat.rhoSR)
        self.rhoFR = df.Constant(ip.param.mat.rhoFR)
        self.gammaFR = df.Constant(ip.param.mat.gammaFR)
        self.R = df.Constant(ip.param.mat.R)
        self.Theta = df.Constant(ip.param.mat.Theta)

        # spatial varying material parameters
        self.kF = ip.param.mat.kF
        self.lambdaS = ip.param.mat.lambdaS
        self.muS = ip.param.mat.muS

        # FEM paramereters and additionals
        self.solver_param = ip.param.fem
        if hasattr(ip.param.add, "prim_vars"):
            self.prim_vars_list.extend(ip.param.add.prim_vars)
            self.ele_types.extend(ip.param.add.ele_types)
            self.ele_orders.extend(ip.param.add.ele_orders)
            self.tensor_order.extend(ip.param.add.tensor_orders)
            self.molFkappa = ip.param.add.molFkappa
            self.DFkappa = ip.param.add.DFkappa
            for idx in range(len(ip.param.add.prim_vars)):
                self.hatrhoFkappa.append(df.Constant(0.0))

        # geometry parameters
        self.mesh = ip.geom.mesh
        self.dim = ip.geom.dim
        self.n_bound = ip.geom.n_bound
        self.d_bound = ip.geom.d_bound

    def set_micro_models(self, prod_terms: list):
        self.prod_terms = prod_terms
        # init growth terms
        self.hatnS = df.Function(self.CG1_sca)
        for idx in range(1, len(prod_terms)):
            if prod_terms[idx] is not None:
                self.hatrhoFkappa[idx-1] = df.Function(self.CG1_sca)
            else:
                self.hatrhoFkappa[idx-1] = df.Constant(0.0)

    def output(self, time) -> None:
        for idx, prim_var in enumerate(self.prim_vars_list):
            write_field2xdmf(self.output_file, self.sol.sub(idx), prim_var, time)
        write_field2xdmf(self.output_file, self.intern_output[0], "hatcFt", time, function_space=self.CG1_sca)  # , self.eval_points, self.mesh)

    def unpack_prim_pvars(self, function_space: df.Function) -> tuple:
        """unpacks primary variables and returns tuple"""
        u = df.split(function_space)
        p = []
        for i in range(self.n_init_prim_vars, len(u)):
            p.append(u[i])
        return u[0], u[1], u[2], p 

    def set_hets_if_needed(self, field):
        if type(field) is float:
            field = df.Constant(field)
        else:
            help_func = field
            field = df.Function(df.FunctionSpace(self.mesh, "DG", 0))
            #field.interpolate(InitialDistribution(help_func))
            field.interpolate(help_func)
        return field

    def set_heterogenities(self):
        self.kF = self.set_hets_if_needed(self.kF)
        self.lambdaS = self.set_hets_if_needed(self.lambdaS)
        self.muS = self.set_hets_if_needed(self.muS)
        for i in range(len(self.DFkappa)):
            self.DFkappa[i] = self.set_hets_if_needed(self.DFkappa[i])

    def set_weak_form(self) -> None:
        ##############################################################################
        # Get Ansatz and test functions
        self.sol_old = df.Function(self.function_space)  # old primaries
        u, p, nS, cFkappa = self.unpack_prim_pvars(self.ansatz_functions)
        _u, _p, _nS, _cFkappa = self.unpack_prim_pvars(self.test_functions)
        u_n, p_n, nS_n, cFkappa_n = self.unpack_prim_pvars(self.sol_old)

        ##############################################################################
        # Calculate volume fractions
        rhoS = nS * df.Constant(self.rhoSR)
        nF = 1.0 - nS

        ##############################################################################
        # Get growth terms
        hatnS = self.hatnS
        hatrhoS = self.hatnS * df.Constant(self.rhoSR) 
        hatrhoF = - hatrhoS

        ##############################################################################
        # Kinematics
        I = ufl.Identity(len(u))
        F_S = I + ufl.grad(u)
        F_Sn = I + ufl.grad(u_n)
        J_S = ufl.det(F_S)
        C_S = F_S.T * F_S
        B_S = F_S * F_S.T
        dF_Sdt = (F_S - F_Sn) / df.Constant(self.dt)
        L_S = dF_Sdt * ufl.inv(F_S)
        D_S = (L_S + L_S.T) / 2.0
        ##############################################################################
        # Rodriguez Split
        if self.flag_defSplit:
            time = df.Constant(0)
            J_Sg = ufl.exp(hatnS / nS_n * time)
            F_Sg = (J_Sg ** (1 / self.dim)) * I
            F_Se = F_S * ufl.inv(F_Sg)
            J_Se = ufl.det(F_Se)
            B_Se = F_Se * F_Se.T
        else:
            B_Se = B_S
            J_Se = J_S
        ##############################################################################
        # Calculate velocities
        v = (u - u_n) / df.Constant(self.dt)
        div_v = ufl.inner(D_S, I)
        dnSdt = (nS - nS_n) / df.Constant(self.dt)
        dcFkappadt = []
        for i, cFk in enumerate(cFkappa):
            dcFkappadt.append((cFk - cFkappa_n[i]) / df.Constant(self.dt))

        ##############################################################################
        # Calculate Stress
        lambdaS = df.Constant(self.lambdaS)
        muS = df.Constant(self.muS)

        TS_E = (muS * (B_Se - I) + lambdaS * ufl.ln(J_Se) * I) / J_Se
        T = TS_E - p * I
        P = J_S * T * ufl.inv(F_S.T)

        ##############################################################################
        # Define weak forms
        kD = df.Constant(self.kF) / df.Constant(self.gammaFR)
        dx = df.Measure("dx", domain=self.mesh)
        ##############################################################################
        # Momentum balance of overall aggregate
        res_LMo1 = ufl.inner(P, ufl.grad(_u)) * dx
        res_LMo2 = + J_S * kD / (nF*nF) * hatrhoF * hatrhoF * ufl.dot(ufl.dot(v, ufl.inv(F_S)), _u) * dx
        res_LMo3 = - J_S * kD / nF * ufl.dot(ufl.dot(ufl.grad(p), ufl.inv(F_S)), _u) * dx
        res_LMo = res_LMo1 + res_LMo2 + res_LMo3

        ##############################################################################
        # Volume balance of the mixture
        res_VBm1 = J_S * div_v * _p * dx 
        res_VBm2 = - J_S * (hatrhoS / rhoS + hatrhoF / self.rhoFR) * _p * dx
        res_VBm31 = ufl.dot(ufl.grad(p), ufl.inv(C_S)) + hatrhoF / nF * ufl.dot(v, ufl.inv(F_S.T))
        res_VBm3 = J_S * kD * ufl.inner(res_VBm31, ufl.grad(_p)) * dx
        res_VBm = res_VBm1 + res_VBm2 + res_VBm3

        ##############################################################################
        # Volume balance of solid body
        res_VB1 = J_S * (dnSdt - self.hatnS) * _nS * dx
        res_VB2 = J_S * nS * div_v * _nS * dx
        res_VB = res_VB1 + res_VB2

        ##############################################################################
        # Concentration balance of additionals
        res_CBkappa = []
        for i, cFk in enumerate(cFkappa):
            dFkappa = self.DFkappa[i] / (self.R * self.Theta)
            diffvelo = dFkappa * (ufl.dot(ufl.grad(cFk), ufl.inv(C_S)) + self.hatrhoFkappa[i] / nF * ufl.dot(v, ufl.inv(F_S.T)))
            seepagevelo = - cFk * kD * (ufl.dot(ufl.grad(p), ufl.inv(C_S)) - self.hatrhoFkappa[i] / nF * ufl.dot(v, ufl.inv(F_S.T)))
            res_CBkappa1 = J_S * (nF * dcFkappadt[i] - self.hatrhoFkappa[i] / df.Constant(self.molFkappa[i])) * _cFkappa[i] * dx
            res_CBkappa2 = J_S * cFk * (div_v - hatrhoS / rhoS) * _cFkappa[i] * dx
            res_CBkappa3 = J_S * ufl.inner(diffvelo, ufl.grad(_cFkappa[i])) * dx
            res_CBkappa4 = J_S * ufl.inner(seepagevelo, ufl.grad(_cFkappa[i])) * dx
            res_CBkappa.append(res_CBkappa1 + res_CBkappa2 + res_CBkappa3 + res_CBkappa4)

        ##############################################################################
        # sum up to total residual
        res_tot = res_LMo + res_VBm + res_VB
        for res_CBk in res_CBkappa:
            res_tot += res_CBk
        if self.n_bound is not None:
            res_tot += self.n_bound

        self.intern_output = [self.hatrhoFkappa[0]]
        self.residuum = res_tot

    def set_solver(self):
        # Make sure quadrature_degree stays at 2
        prm = df.parameters["form_compiler"]
        prm["quadrature_degree"] = 2
        self.sol = self.ansatz_functions
        solver = aux.Solver()
        solver.solver_type = self.solver_param.solver_type
        solver.abs = self.solver_param.abs
        solver.rel = self.solver_param.rel
        solver.maxIter = self.solver_param.maxIter
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
            timer_start = time.time()
            n_iter, converged = self.solver.solve()
            # actualize prod terms
            self.actualize_prod_terms()
            # Output solution
            if out_count >= self.output_interval:
                timer_end = time.time()
                print("Time: {}".format(t), "  ", "Converged in steps: {}".format(n_iter), " ", 
                      "Calculation time: {}".format(timer_end-timer_start))
                out_count = 0.0
                self.output(t)
            # Update history fields
            self.sol_old.assign(self.sol)
