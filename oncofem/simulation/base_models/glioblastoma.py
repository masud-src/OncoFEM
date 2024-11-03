"""
Definition of two-phase material according to the Theory of Porous Media. The solid phase composes of healthy tissue Sh,
active tumor tissue St and necrotic tissue Sn. In the fluid constituent multiple components can be resolved adaptively.
The model is designed for simulation of glioblastoma multiforme (GBM).

Class:
    Glioblastoma:      Derived from BaseModel. See class description for more information.
"""
import time
from typing import Union
from oncofem.helper.fem_aux import InitialCondition, Solver, MapAverageMaterialProperty
import oncofem.helper.general as gen
from oncofem.simulation.problem import Problem
from oncofem.helper.io import write_field2xdmf
import dolfin as df
import ufl
from oncofem.simulation.base_models.base_model import BaseModel

class Glioblastoma(BaseModel):
    """

    *Methods:*
        set_boundaries:         Sets surface boundaries, e. g. Dirichlet and Neumann boundaries.
        assign_if_function:     Helper function, that assigns values to the solution field and old solution field, if a
                                function is given. Used for adaptive initial conditions.
        actualize_prod_terms:   Actualises production terms in each time step.
        set_initial_conditions: Sets initial condition for adaptive system. Can take scalars, distribution from 
                                MeshFunctions and Functions.
        set_function_spaces:    Sets function space for primary variables u, p, nS, cFdelta and for internal .
        set_param:              Sets parameter needed for model class from given problem.
        set_micro_models:       Sets the chosen bio-chemical model set-up on the microscale.
        output:                 Defines the way the output shall be created and what shall be exported.
        unpack_prim_pvars:      Unpacks primary variables and returns tuple.
        set_hets_if_needed:     Sets the heterogenities, if a distribution is given
        set_heterogeneities:    Set heterogenities on the domain.     
        set_weak_form:          Sets the weak form of the system of partial differential equations.  
        set_solver:             Sets up the numerical solver method to solve the weak form.                
        solve:                  Method for solving the particular model within one time interval.
    """
    def __init__(self):
        super().__init__()
        # general info
        self.output_file = None
        self.flag_defSplit = True

        # FEM paramereters
        self.solver_param = None
        self.prim_vars_list = ["u", "p", "nSh", "nSt", "nSn", "cFt"]
        self.n_init_prim_vars = len(self.prim_vars_list)
        self.tensor_order = [1, 0, 0, 0, 0, 0]
        self.ele_types = ["CG", "CG", "CG", "CG", "CG", "CG"]
        self.ele_orders = [2, 1, 1, 1, 1, 1]
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
        self.hatnSh = None
        self.hatnSt = None
        self.hatnSn = None
        self.hatrhoFt = None
        self.hatrhoFkappa = []
        self.bio_terms = []
        self.residuum = None
        self.intern_output = None
        self.solver = None

        # intrinsic material parameters
        self.rhoShR = None
        self.rhoStR = None
        self.rhoSnR = None
        self.rhoFR = None
        self.gammaFR = None
        self.molFt = None
        self.R = None
        self.Theta = None

        # spatial varying material parameters
        self.kF = None
        self.lambdaSh = None
        self.lambdaSt = None
        self.lambdaSn = None
        self.muSh = None
        self.muSt = None
        self.muSn = None
        self.DFt = None

        # initial conditions
        self.uS_0S = None
        self.p_0S = None
        self.nSh_0S = None
        self.nSt_0S = None
        self.nSn_0S = None
        self.nF_0S = None
        self.cFt_0S = None

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

    def set_boundaries(self, d_bound, n_bound) -> None:
        self.d_bound = d_bound
        self.n_bound = n_bound    

    def assign_if_function(self, var, index) -> None:
        if type(var) is df.Function:
            df.assign(self.sol.sub(index), var)
            df.assign(self.sol_old.sub(index), var)

    def actualize_prod_terms(self) -> None:
        if self.prod_terms[0] is not None:
            self.hatnSh.assign(df.project(self.prod_terms[0], self.CG1_sca))
        else:
            self.hatnSh = df.Constant(0.0)

        if self.prod_terms[1] is not None:
            self.hatnSt.assign(df.project(self.prod_terms[1], self.CG1_sca))
        else:
            self.hatnSt = df.Constant(0.0)

        if self.prod_terms[2] is not None:
            self.hatnSn.assign(df.project(self.prod_terms[2], self.CG1_sca))
        else:
            self.hatnSn = df.Constant(0.0)

        if self.prod_terms[3] is not None:
            self.hatrhoFt.assign(df.project(self.prod_terms[3], self.CG1_sca))
        else:
            self.hatrhoFt = df.Constant(0.0)

        for idx in range(4, len(self.prod_terms)):
            if self.prod_terms[idx] is not None:
                self.hatrhoFkappa[idx-4].assign(df.project(self.prod_terms[idx], self.CG1_sca))
            else:
                self.hatrhoFkappa[idx-4] = df.Constant(0.0)

    def set_initial_conditions(self, init, add) -> None:
        """
        Sets initial condition for adaptive system. Can take scalars, distribution from MeshFunctions and Functions.
        """
        # set intern vars
        self.uS_0S = init.uS_0S
        self.p_0S = init.p_0S
        self.nSh_0S = init.nSh_0S
        self.nSt_0S = init.nSt_0S
        self.nSn_0S = init.nSn_0S
        self.cFt_0S = init.cFt_0S
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
        init_set.append(gen.check_if_type(init.nSh_0S, df.Function, None))
        init_set.append(gen.check_if_type(init.nSt_0S, df.Function, None))
        init_set.append(gen.check_if_type(init.nSn_0S, df.Function, None))
        init_set.append(gen.check_if_type(init.cFt_0S, df.Function, None))
        if self.cFkappa_0S is not None:
            for cFd_0S in self.cFkappa_0S:
                init_set.append(gen.check_if_type(cFd_0S, df.Function, None))
        self.sol.interpolate(InitialCondition(init_set))
        self.sol_old.interpolate(InitialCondition(init_set))

        self.assign_if_function(self.uS_0S, 0)
        self.assign_if_function(self.p_0S, 1)
        self.assign_if_function(self.nSh_0S, 2)
        self.assign_if_function(self.nSt_0S, 3)
        self.assign_if_function(self.nSn_0S, 4)
        self.assign_if_function(self.cFt_0S, 5)
        if self.cFkappa_0S is not None:
            for idx, cFd_0S in enumerate(self.cFkappa_0S):
                self.assign_if_function(cFd_0S, 5+idx)
        # production terms
        self.actualize_prod_terms()

    def set_function_spaces(self):
        """
            sets function space for primary variables u, p, cIn, cIt, cIv and for internal variables
        """
        elements = []
        for idx, e_type in enumerate(self.ele_types):
            if self.tensor_order[idx] == 0:
                ele = df.FiniteElement(e_type, self.mesh.ufl_cell(), self.ele_orders[idx])
            elif self.tensor_order[idx] == 1:
                ele = df.VectorElement(e_type, self.mesh.ufl_cell(), self.ele_orders[idx])
            elif self.tensor_order[idx] == 2:
                ele = df.TensorElement(e_type, self.mesh.ufl_cell(), self.ele_orders[idx])
            elements.append(ele)
        self.finite_element = ufl.MixedElement(elements)
        self.function_space = df.FunctionSpace(self.mesh, self.finite_element)
        self.DG0 = df.FunctionSpace(self.mesh, "DG", 0)
        self.CG1_sca = df.FunctionSpace(self.mesh, "CG", 1)
        self.CG1_vec = df.VectorFunctionSpace(self.mesh, "CG", 1)
        self.CG1_ten = df.TensorFunctionSpace(self.mesh, "CG", 1)
        self.ansatz_functions = df.Function(self.function_space)
        self.test_functions = df.TestFunction(self.function_space)

    def set_param(self, ip:Problem) -> None:
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
        self.rhoShR = df.Constant(ip.param.mat.rhoShR)
        self.rhoStR = df.Constant(ip.param.mat.rhoStR)
        self.rhoSnR = df.Constant(ip.param.mat.rhoSnR)
        self.rhoFR = df.Constant(ip.param.mat.rhoFR)
        self.gammaFR = df.Constant(ip.param.mat.gammaFR)
        self.molFt = df.Constant(ip.param.mat.molFt)
        self.R = df.Constant(ip.param.mat.R)
        self.Theta = df.Constant(ip.param.mat.Theta)

        # spatial varying material parameters
        self.kF = ip.param.mat.kF
        self.lambdaSh = ip.param.mat.lambdaSh
        self.lambdaSt = ip.param.mat.lambdaSt
        self.lambdaSn = ip.param.mat.lambdaSn
        self.muSh = ip.param.mat.muSh
        self.muSt = ip.param.mat.muSt
        self.muSn = ip.param.mat.muSn
        self.DFt = ip.param.mat.DFt

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

    def set_process_models(self, prod_terms:list) -> None:
        self.prod_terms = prod_terms
        # init growth terms
        self.hatnSh = df.Function(self.CG1_sca)
        self.hatnSt = df.Function(self.CG1_sca)
        self.hatnSn = df.Function(self.CG1_sca)
        self.hatrhoFt = df.Function(self.CG1_sca)
        for idx in range(4, len(prod_terms)):
            if prod_terms[idx] is not None:
                self.hatrhoFkappa[idx-4] = df.Function(self.CG1_sca)
            else:
                self.hatrhoFkappa[idx-4] = df.Constant(0.0)

    def output(self, time_step:float) -> None:
        for idx, prim_var in enumerate(self.prim_vars_list):
            write_field2xdmf(self.output_file, self.sol.sub(idx), prim_var, time_step)
        write_field2xdmf(self.output_file, self.intern_output[0], "nF", time_step, function_space=self.CG1_sca)  # , self.eval_points, self.mesh)
        write_field2xdmf(self.output_file, self.intern_output[1], "hatrhoSh", time_step, function_space=self.CG1_sca)  # , self.eval_points, self.mesh)
        write_field2xdmf(self.output_file, self.intern_output[2], "hatrhoSt", time_step, function_space=self.CG1_sca)  # , self.eval_points, self.mesh)
        write_field2xdmf(self.output_file, self.intern_output[3], "hatrhoSn", time_step, function_space=self.CG1_sca)  # , self.eval_points, self.mesh)
        write_field2xdmf(self.output_file, self.intern_output[4], "hatrhoFt", time_step, function_space=self.CG1_sca)  # , self.eval_points, self.mesh)
        write_field2xdmf(self.output_file, self.intern_output[5], "hatrhoFn", time_step, function_space=self.CG1_sca)  # , self.eval_points, self.mesh)
        write_field2xdmf(self.output_file, self.intern_output[6], "TS_E", time_step, function_space=self.CG1_ten)  # , self.eval_points, self.mesh)
        write_field2xdmf(self.output_file, self.intern_output[7], "div_v", time_step, function_space=self.CG1_sca)  # , self.eval_points, self.mesh)

    def unpack_prim_pvars(self, function_space: df.Function) -> tuple:
        """unpacks primary variables and returns tuple"""
        u = df.split(function_space)
        p = []
        for i in range(self.n_init_prim_vars, len(u)):
            p.append(u[i])
        return u[0], u[1], u[2], u[3], u[4], u[5], p

    def set_hets_if_needed(self, field: Union[float, MapAverageMaterialProperty]) -> Union[df.Constant, df.Function]:
        if type(field) is float:
            field = df.Constant(field)
        else:
            help_func = field
            field = df.Function(df.FunctionSpace(self.mesh, "DG", 0))
            field.interpolate(help_func)
        return field

    def set_structural_parameters(self) -> None:
        self.kF = self.set_hets_if_needed(self.kF)
        self.lambdaSh = self.set_hets_if_needed(self.lambdaSh)
        self.lambdaSt = self.set_hets_if_needed(self.lambdaSt)
        self.lambdaSn = self.set_hets_if_needed(self.lambdaSn)
        self.muSh = self.set_hets_if_needed(self.muSh)
        self.muSt = self.set_hets_if_needed(self.muSt)
        self.muSn = self.set_hets_if_needed(self.muSn)
        self.DFt = self.set_hets_if_needed(self.DFt)
        for i in range(len(self.DFkappa)):
            self.DFkappa[i] = self.set_hets_if_needed(self.DFkappa[i])

    def set_weak_form(self) -> None:
        ##############################################################################
        # Get Ansatz and test functions
        #######################################
        self.sol_old = df.Function(self.function_space)  # old primaries
        u, p, nSh, nSt, nSn, cFt, cFkappa = self.unpack_prim_pvars(self.ansatz_functions)
        _u, _p, _nSh, _nSt, _nSn, _cFt, _cFkappa = self.unpack_prim_pvars(self.test_functions)
        u_n, p_n, nSh_n, nSt_n, nSn_n, cFt_n, cFkappa_n = self.unpack_prim_pvars(self.sol_old)

        ##############################################################################
        # Calculate volume fractions
        #######################################
        nS = nSh + nSt + nSn
        rhoS = (nSh * df.Constant(self.rhoShR) + nSt * df.Constant(self.rhoStR) + nSn * df.Constant(self.rhoSnR)) / nS
        nF = 1.0 - nS

        ##############################################################################
        # Get growth terms
        #######################################
        hatnS = self.hatnSh + self.hatnSt + self.hatnSn
        hatrhoS = (self.hatnSh * df.Constant(self.rhoShR) + self.hatnSt * df.Constant(self.rhoStR) + self.hatnSn * df.Constant(self.rhoSnR)) / nS
        hatrhoF = - hatrhoS

        ##############################################################################
        # Kinematics
        #######################################
        I = ufl.Identity(len(u))
        F_S = I + ufl.grad(u)
        F_Sn = I + ufl.grad(u_n)
        J_S = ufl.det(F_S)
        C_S = F_S.T * F_S
        B_S = F_S * F_S.T
        dF_Sdt = (F_S - F_Sn) / df.Constant(self.dt)
        L_S = dF_Sdt * ufl.inv(F_S)
        D_S = (L_S + L_S.T) / 2.0
        #######################################
        # Rodriguez Split
        #######################################
        if self.flag_defSplit:
            nS_n = nSh_n + nSt_n + nSn_n
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
        # Time-dependent fields
        self.time = df.Constant(0.0)
        #######################################
        # Calculate velocity
        #######################################
        v = (u - u_n) / df.Constant(self.dt)
        div_v = ufl.inner(D_S, I)
        #######################################
        # Calculate storage terms
        #######################################
        dnShdt = (nSh - nSh_n) / df.Constant(self.dt)
        dnStdt = (nSt - nSt_n) / df.Constant(self.dt)
        dnSndt = (nSn - nSn_n) / df.Constant(self.dt)
        dcFtdt = (cFt - cFt_n) / df.Constant(self.dt)
        dcFkappadt = []
        for i, cFk in enumerate(cFkappa):
            dcFkappadt.append((cFk - cFkappa_n[i]) / df.Constant(self.dt))
        ##############################################################################

        ##############################################################################
        # Calculate Stress
        #######################################
        lambdaS = (df.Constant(self.lambdaSh) * nSh + df.Constant(self.lambdaSt) * nSt + df.Constant(self.lambdaSn) * nSn) / (nSh + nSt + nSn)
        muS = (df.Constant(self.muSh) * nSh + df.Constant(self.muSt) * nSt + df.Constant(self.muSn) * nSn) / (nSh + nSt + nSn)

        TS_E = (muS * (B_Se - I) + lambdaS * ufl.ln(J_Se) * I) / J_Se
        T = TS_E - p * I
        P = J_S * T * ufl.inv(F_S.T)
        ##############################################################################

        ##############################################################################
        # Define weak forms
        #######################################
        kD = df.Constant(self.kF) / df.Constant(self.gammaFR)
        dFt = self.DFt / (self.R * self.Theta)
        dx = df.Measure("dx", domain=self.mesh)
        #######################################
        # Momentum balance of overall aggregate
        res_LMo1 = ufl.inner(P, ufl.grad(_u)) * dx
        res_LMo2 = + J_S * kD / (nF*nF) * hatrhoF * hatrhoF * ufl.dot(ufl.dot(v, ufl.inv(F_S)), _u) * dx
        res_LMo3 = - J_S * kD / nF * ufl.dot(ufl.dot(ufl.grad(p), ufl.inv(F_S)), _u) * dx
        res_LMo = res_LMo1 + res_LMo2 + res_LMo3
        #######################################

        #######################################
        # Volume balance of the mixture
        res_VBm1 = J_S * div_v * _p * dx 
        res_VBm2 = - J_S * (hatrhoS / rhoS + hatrhoF / self.rhoFR) * _p * dx
        res_VBm31 = ufl.dot(ufl.grad(p), ufl.inv(C_S)) + hatrhoF / nF * ufl.dot(v, ufl.inv(F_S.T))
        res_VBm3 = J_S * kD * ufl.inner(res_VBm31, ufl.grad(_p)) * dx
        res_VBm = res_VBm1 + res_VBm2 + res_VBm3
        #######################################

        #######################################
        # Volume balance of healthy cells
        res_VBh1 = J_S * (dnShdt - self.hatnSh) * _nSh * dx
        res_VBh2 = J_S * nSh * div_v * _nSh * dx
        res_VBh = res_VBh1 + res_VBh2
        #######################################

        #######################################
        # Volume balance of tumor cells
        res_VBt1 = J_S * (dnStdt - self.hatnSt) * _nSt * dx
        res_VBt2 = J_S * nSt * div_v * _nSt * dx
        res_VBt = res_VBt1 + res_VBt2
        #######################################

        #######################################
        # Volume balance of necrotic cells
        res_VBn1 = J_S * (dnSndt - self.hatnSn) * _nSn * dx
        res_VBn2 = J_S * nSn * div_v * _nSn * dx
        res_VBn = res_VBn1 + res_VBn2
        #######################################

        #######################################
        # Concentration balance of solved cancer cells
        nFcFtw_Ft1 = dFt * (ufl.dot(ufl.grad(cFt), ufl.inv(C_S)) + self.hatrhoFt / nF * ufl.dot(v, ufl.inv(F_S.T))) 
        nFcFtw_Ft2 = - cFt * kD * (ufl.dot(ufl.grad(p), ufl.inv(C_S)) + self.hatrhoFt / nF * ufl.dot(v, ufl.inv(F_S.T)))
        nFcFtw_Ft = nFcFtw_Ft1 + nFcFtw_Ft2
        res_CBt1 = J_S * (nF * dcFtdt - self.hatrhoFt / df.Constant(self.molFt)) * _cFt * dx
        res_CBt2 = J_S * cFt * (div_v - hatrhoS / rhoS) * _cFt * dx
        res_CBt3 = J_S * ufl.inner(nFcFtw_Ft, ufl.grad(_cFt)) * dx
        res_CBt = res_CBt1 + res_CBt2 + res_CBt3
        #######################################

        #######################################
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
        #######################################

        # sum up to total residual
        res_tot = res_LMo + res_VBm + res_VBh + res_VBt + res_VBn + res_CBt
        for res_CBk in res_CBkappa:
            res_tot += res_CBk
        if self.n_bound is not None:
            res_tot += self.n_bound
        ##############################################################################
        self.intern_output = [nF, self.hatnSh, self.hatnSt, self.hatnSn, self.hatrhoFt, self.hatrhoFkappa[0], TS_E, div_v]
        self.residuum = res_tot

    def set_solver(self) -> None:
        # Make sure quadrature_degree stays at 2
        prm = df.parameters["form_compiler"]
        prm["quadrature_degree"] = 2
        self.sol = self.ansatz_functions
        solver = Solver()
        solver.solver_type = self.solver_param.solver_type
        solver.abs = self.solver_param.abs
        solver.rel = self.solver_param.rel
        solver.maxIter = self.solver_param.maxIter
        self.solver = solver.set_non_lin_solver(self.residuum, self.sol, self.d_bound)

    def solve(self) -> None:
        # Initialize  and time loop
        t = 0.0
        out_count = 0.0
        time_flag = True
        self.output(t)
        print("Initial step is written")
        while t < self.T_end:
            # Increment solution time
            t = t + self.dt
            self.time.assign(t)
            out_count += self.dt
            # Calculate current solution
            if time_flag:
                timer_start = time.time()
                time_flag = False
            n_iter, converged = self.solver.solve()
            # actualize prod terms
            self.actualize_prod_terms()
            # Output solution
            if out_count >= self.output_interval:
                timer_end = time.time()
                time_flag = True
                print("Time: {}".format(t), "  ", "Converged in steps: {}".format(n_iter), " ",
                      "Calculation time: {:.2f}".format(timer_end - timer_start),
                      "finish_meter: {:.2f}".format(t / self.T_end))
                out_count = 0.0
                self.output(t)
            # Update history fields
            self.sol_old.assign(self.sol)
