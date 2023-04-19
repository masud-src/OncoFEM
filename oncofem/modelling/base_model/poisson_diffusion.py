"""
Definition of two phase material with mass exchange

Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""
import oncofem.helper.solver as solv
import oncofem.helper.general as gen
from oncofem.struc.problem import Problem
from oncofem.helper.io import write_field2xdmf
import dolfin as df
import ufl
from oncofem.modelling.base_model.base_model import BaseModel
from oncofem.modelling.base_model.base_model import InitialDistribution, InitialCondition


class PoissonDiffusion(BaseModel):
    """
    t.b.d.
    """
    def __init__(self):
        super().__init__()
        # general info
        self.output_file = None

        # FEM parameters
        self.solver_param = None
        self.prim_vars_list = ["u"]
        self.n_init_prim_vars = len(self.prim_vars_list)
        self.tensor_order = [0]
        self.ele_types = ["CG"]
        self.ele_orders = [1]
        self.finite_element = None
        self.function_space = None
        self.ansatz_functiona = None
        self.test_functiona = None
        self.DG0_sca = None
        self.CG1_sca = None
        self.CG1_vec = None
        self.CG1_ten = None
        self.sol = None
        self.sol_old = None

        # geometry parameters
        self.mesh = None
        self.dim = None
        self.n_bound = None
        self.d_bound = None

        # weak form, output and solver
        self.hatrhoS = None
        self.bio_terms = None
        self.residuum = None
        self.intern_output = None
        self.solver = None

        # intrinsic material parameters
        self.rhoSR = None
        self.rhoFR = None
        self.gammaFR = None

        # spatial varying material parameters
        self.kF = None
        self.lambdaS = None
        self.muS = None

        # initial conditions
        self.uS_0S = None
        self.p_0S = None
        self.nS_0S = None

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
        if self.hatrhoS is not None:
            self.hatrhoS.assign(df.project(self.hatrhoS, self.CG1_sca))
        else:
            self.hatrhoS = df.Constant(0.0)

    def set_initial_conditions(self, init, add):
        """
        Sets initial condition for adaptive system. Can take scalars, distribution from MeshFunctions and Functions.
        """
        # set intern vars
        self.uS_0S = init.uS_0S
        self.p_0S = init.p_0S
        self.nS_0S = init.nSh_0S

        # collect for interpolation
        init_set = gen.check_if_type(init.uS_0S, df.Function, None)
        init_set.append(gen.check_if_type(init.p_0S, df.Function, None))
        init_set.append(gen.check_if_type(init.nS_0S, df.Function, None))
        self.sol.interpolate(InitialCondition(init_set))
        self.sol_old.interpolate(InitialCondition(init_set))

        self.assign_if_function(self.uS_0S, 0)
        self.assign_if_function(self.p_0S, 1)
        self.assign_if_function(self.nS_0S, 2)

        # production terms
        self.actualize_prod_terms()

    def set_function_spaces(self):
        """
        sets function space for primary variables u, p, nS and for internal variables
        """
        element_u = df.VectorElement(self.type_u, self.mesh.ufl_cell(), self.order_u)
        element_p = df.FiniteElement(self.type_p, self.mesh.ufl_cell(), self.order_p)
        element_nS = df.FiniteElement(self.type_nS, self.mesh.ufl_cell(), self.order_nS)
        self.finite_element = df.MixedElement(element_u, element_p, element_nS)
        self.function_space = df.FunctionSpace(self.mesh, self.finite_element)
        self.DG0_sca = df.FunctionSpace(self.mesh, "DG", 0)
        self.CG1_sca = df.FunctionSpace(self.mesh, "CG", 1)
        self.CG1_vec = df.VectorFunctionSpace(self.mesh, "CG", 1)
        self.CG1_ten = df.TensorFunctionSpace(self.mesh, "CG", 1)
        self.ansatz_functions = df.Function(self.function_space)
        self.test_functions = df.TestFunction(self.function_space)

    def set_param(self, input):
        self.eval_points = input.param.gen.eval_points
        self.output_file = input.param.gen.output_file
        self.type_u = input.param.fem.type_u
        self.type_p = input.param.fem.type_p
        self.type_nS = input.param.fem.type_nS
        self.order_u = input.param.fem.order_u
        self.order_p = input.param.fem.order_p
        self.order_nS = input.param.fem.order_nS
        self.mesh = input.geom.mesh
        self.domain = input.geom.domain
        self.growthArea = input.geom.growthArea
        self.dx = input.geom.dx
        self.n_bound = input.geom.n_bound
        self.d_bound = input.geom.d_bound
        self.hatrhoS = df.Constant(input.param.mat.hatrhoS)
        self.rhoSR = df.Constant(input.param.mat.rhoSR)
        self.rhoFR = df.Constant(input.param.mat.rhoFR)
        self.lambdaS = df.Constant(input.param.mat.lambdaS)
        self.muS = df.Constant(input.param.mat.muS)
        self.gammaFR = df.Constant(input.param.mat.gammaFR)
        self.nS_0S = df.Constant(input.param.mat.nS_0S)
        self.kF_0S = df.Constant(input.param.mat.kF_0S)
        self.solver_param = input.param.fem.solver_param
        self.dt = input.param.time.dt
        self.T_end = input.param.time.T_end

    def set_bio_chem_models(self, ip):
        pass

    def output(self, time) -> None:
        for idx, prim_var in enumerate(self.prim_vars_list):
            write_field2xdmf(self.output_file, self.sol.sub(idx), prim_var, time)
        write_field2xdmf(self.output_file, self.intern_output[0], "nF", time, function_space=self.CG1_sca)  # , self.eval_points, self.mesh)
        write_field2xdmf(self.output_file, self.intern_output[1], "hatrhoS", time, function_space=self.CG1_sca)  # , self.eval_points, self.mesh)
        write_field2xdmf(self.output_file, self.intern_output[2], "TS_E", time, function_space=self.CG1_ten)  # , self.eval_points, self.mesh)
        write_field2xdmf(self.output_file, self.intern_output[3], "div_v", time, function_space=self.CG1_sca)  # , self.eval_points, self.mesh)

    def set_hets_if_needed(self, field):
        pass

    def set_heterogenities(self):
        pass

    def set_weak_form(self) -> None:
        ##############################################################################
        # Get Ansatz and test functions
        #######################################
        self.sol_old = df.Function(self.function_space)  # old primaries
        u, p, nS = df.split(self.ansatz_functions)
        _u, _p, _nS = df.split(self.test_functions)
        u_n, p_n, nS_n = df.split(self.sol_old)

        # Calculate volume fractions
        nF = 1.0 - nS
        hatnS = self.hatrhoS / self.rhoSR
        hatrhoF = - self.hatrhoS
        hatnF = hatrhoF / self.rhoFR

        # Calculate kinematics
        I = ufl.Identity(len(u))
        F_S = I + ufl.grad(u)
        F_Sn = I + ufl.grad(u_n)
        C_S = F_S.T * F_S
        J_S = ufl.det(F_S)
        B_S = F_S * F_S.T

        # Calculate velocity and time dependent variables
        dF_Sdt = (F_S - F_Sn) / self.dt
        L_S = dF_Sdt * ufl.inv(F_S)
        D_S = (L_S + L_S.T) / 2.0
        div_v = ufl.inner(D_S, ufl.Identity(len(u)))
        v = (u - u_n) / self.dt
        dnSdt = (nS - nS_n) / self.dt

        # Calculate Stresses
        TS_E = self.muS * (B_S - I) + self.lambdaS * J_S * I
        T = TS_E - p * df.Identity(len(u))
        P = J_S * T * ufl.inv(F_S.T)

        ##############################################################################
        # Define weak forms
        #######################################
        kD = df.Constant(self.kF) / df.Constant(self.gammaFR)
        dx = df.Measure("dx", domain=self.mesh)
        #######################################
        # Momentum balance of overall aggregate
        res_LMo1 = ufl.inner(P, ufl.grad(_u)) * dx
        res_LMo2 = + J_S * kD / (nF * nF) * hatrhoF * hatrhoF * ufl.dot(ufl.dot(v, ufl.inv(F_S)), _u) * dx
        res_LMo3 = - J_S * kD / nF * ufl.dot(ufl.dot(ufl.grad(p), ufl.inv(F_S)), _u) * dx
        res_LMo = res_LMo1 + res_LMo2 + res_LMo3
        #######################################

        #######################################
        # Volume balance of the mixture
        res_VBm1 = J_S * div_v * _p * dx
        res_VBm2 = - J_S * (self.hatrhoS / self.rhoSR + hatrhoF / self.rhoFR) * _p * dx
        res_VBm31 = ufl.dot(ufl.grad(p), ufl.inv(C_S)) + hatrhoF / nF * ufl.dot(v, ufl.inv(F_S.T))
        res_VBm3 = J_S * kD * ufl.inner(res_VBm31, ufl.grad(_p)) * dx
        res_VBm = res_VBm1 + res_VBm2 + res_VBm3
        #######################################

        #######################################
        # Volume balance of healthy cells
        res_VB1 = J_S * (dnSdt - self.hatnS) * _nS * dx
        res_VB2 = J_S * nS * div_v * _nS * dx
        res_VB = res_VB1 + res_VB2
        #######################################

        # sum up to total residual
        res_tot = res_LMo + res_VBm + res_VB
        if self.n_bound is not None:
            res_tot += self.n_bound
        ##############################################################################
        self.intern_output = [nF, TS_E, div_v]
        self.residuum = res_tot

    def set_solver(self):
        prm = df.parameters["form_compiler"]
        prm["quadrature_degree"] = 2
        self.sol = self.ansatz_functions
        self.solver = solv.nonlinvarsolver(self.residuum, self.sol, self.d_bound, self.solver_param)

    def solve(self):
        # Store history values for time integration
        w_n = dolfin.Function(self.function_space)
        hatrhoS = dolfin.Function(self.V0)
        time = dolfin.Constant(0)
        u_n, p_n, nS_n = dolfin.split(w_n)

        # Get Ansatz and test functions
        u, p, nS = dolfin.split(self.ansatz_function)
        _u, _p, _nS = dolfin.split(self.test_function)

        # Integration over domain
        dx = self.dx

        # Calculate volume fractions
        nF = 1.0 - nS
        hatnS = hatrhoS / self.rhoSR
        hatrhoF = - hatrhoS
        hatnF = hatrhoF / self.rhoFR

        # Calculate kinematics
        if self.flag_defSplit == True:
            J_SG = dolfin.exp(hatnS / nS_n * time)
        else:
            J_SG = 1.0

        I = ufl.Identity(len(u))
        F_SG = J_SG ** (1 / len(u)) * I
        F_S = I + ufl.grad(u)
        C_S = F_S.T * F_S
        J_S = ufl.det(F_S)
        F_Sn = I + ufl.grad(u_n)
        F_SE = F_S * ufl.inv(F_SG)
        J_SE = ufl.det(F_SE)
        B_SE = F_SE * F_SE.T

        # Calculate velocity and time dependent variables
        dF_Sdt = (F_S - F_Sn) / self.dt
        L_S = dF_Sdt * ufl.inv(F_S)
        D_S = (L_S + L_S.T) / 2.0
        div_v = ufl.inner(D_S, ufl.Identity(len(u)))
        v = (u - u_n) / self.dt
        dnSdt = (nS - nS_n) / self.dt

        # Calculate Stresses
        TS_E = self.muS * (B_SE - I) + self.lambdaS * J_SE * I
        T = TS_E - p * dolfin.Identity(len(u))

        # Calculate seepage-velocity (w_FS)
        cond = ufl.sqrt(hatnF * hatnF)
        numerator = self.kF_0S * nF * nF
        denominator = self.gammaFR * nF * nF + self.kF_0S * hatnF * self.rhoFR
        kappa_FS = dolfin.conditional(cond > dolfin.DOLFIN_EPS, numerator / denominator, self.kF_0S / self.gammaFR)

        # Define weak forms
        res_LMo1 = ufl.inner(ufl.grad(_u), T * J_S * ufl.inv(F_S.T)) * dx
        res_LMo2 = - hatrhoF * J_S / nF * kappa_FS * ufl.inner(ufl.dot(ufl.grad(p), ufl.inv(F_S)) + hatnF / nF * self.rhoFR * v, _u) * dx
        res_LMo = res_LMo1 + res_LMo2

        res_MMo1 = J_S * div_v * _p * dx
        res_MMo2u = ufl.dot(ufl.grad(p), ufl.inv(C_S)) + hatnF / nF * self.rhoFR * ufl.dot(v, ufl.inv(F_S.T))
        res_MMo2 = - J_S * kappa_FS * ufl.inner(res_MMo2u, ufl.grad(_p)) * dx
        res_MMo3 = - J_S * hatnS * (1 - self.rhoSR / self.rhoFR) * _p * dx

        res_MMo = res_MMo1 + res_MMo2 + res_MMo3

        res_MMs1 = J_S * (dnSdt + hatnS) * _nS * dx
        res_MMs2 = nS * J_S * div_v * _nS * dx

        res_MMs = res_MMs1 + res_MMs2

        if self.n_bound is not None:
            res_tot = res_LMo + res_MMo + res_MMs + self.n_bound
        else:
            res_tot = res_LMo + res_MMo + res_MMs

        # Define problem solution
        w = self.ansatz_function
        solver = solv(res_tot, w, self.d_bound, self.solver_param)

        # Set initial conditions
        w_n.interpolate(self.initial_condition)
        w.interpolate(self.initial_condition)

        # Initialize solution time
        t = 0

        output(w, t)

        # Time loop
        while t < self.T_end:
            # Increment solution time
            t += self.dt
            time.assign(t)

            # Calculate current solution
            solver.solve()
            # print("Time: {}".format(t), "  ", "Converged in steps: {}".format(n_iter))

            # Output solution
            output(w, t)

            w_n.assign(w)
