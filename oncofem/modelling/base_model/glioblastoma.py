"""
# **************************************************************************#
#                                                                           #
# === Glioblastoma =========================================================#
#                                                                           #
# **************************************************************************#
# Definition of Glioblastoma 
#
# Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
#
# --------------------------------------------------------------------------#
"""

import oncofem.modelling.base_model.solver as solv
import oncofem.helper.general as gen
from oncofem.struc.problem import Problem
from oncofem.helper.io import write_field2xdmf
import dolfin as df
import ufl
from oncofem.modelling.base_model.base_model import BaseModel
from oncofem.modelling.base_model.base_model import InitialDistribution, InitialCondition

#############################################################
#                                                           #
#  Helper functions                                         #
#                                                           #
#############################################################
class Glioblastoma(BaseModel):

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
        self.hatrhoFdelta = []
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

        # metastatic switch
        self.cFt_ms = None
        self.nSt_ms = None

        # spatial varying material parameters
        self.nSh_0S = None
        self.nSt_0S = None
        self.nSn_0S = None
        self.cFt_0S = None
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
        self.cFdelta = None
        self.molFdelta = None
        self.DFdelta = None
        self.cFdelta_0S = None

        # time parameters
        self.time = None
        self.T_end = None
        self.output_interval = None
        self.dt = None

    def set_boundaries(self, d_bound, n_bound):
        self.d_bound = d_bound
        self.n_bound = n_bound    

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
        self.rhoShR = df.Constant(ip.param.mat.rhoShR)
        self.rhoStR = df.Constant(ip.param.mat.rhoStR)
        self.rhoSnR = df.Constant(ip.param.mat.rhoSnR)
        self.rhoFR = df.Constant(ip.param.mat.rhoFR)
        self.gammaFR = df.Constant(ip.param.mat.gammaFR)
        self.molFt = df.Constant(ip.param.mat.molFt)

        # metastatic switch
        self.cFt_ms = df.Constant(ip.param.mat.cFt_ms)
        self.nSt_ms = df.Constant(ip.param.mat.nSt_ms)

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
        self.solver_param = ip.param.fem.solver_param
        if hasattr(ip.param.add, "prim_vars"):
            self.prim_vars_list.extend(ip.param.add.prim_vars)
            self.ele_types.extend(ip.param.add.ele_types)
            self.ele_orders.extend(ip.param.add.ele_orders)
            self.tensor_order.extend(ip.param.add.tensor_orders)
            self.molFdelta = ip.param.add.molFdelta
            self.DFdelta = ip.param.add.DFdelta
            for idx in range(len(ip.param.add.prim_vars)):
                self.hatrhoFdelta.append(df.Constant(0.0))

        # geometry parameters
        self.mesh = ip.geom.mesh
        self.dim = ip.geom.dim
        self.n_bound = ip.geom.n_bound
        self.d_bound = ip.geom.d_bound

    def set_function_spaces(self):
        """
            sets function space for primary variables u, p, cIn, cIt, cIv and for internal variables
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

    def set_bio_chem_models(self, prod_terms: list):
        self.prod_terms = prod_terms
        # init growth terms
        self.hatnSh = df.Function(self.CG1_sca)
        self.hatnSt = df.Function(self.CG1_sca)
        self.hatnSn = df.Function(self.CG1_sca)
        self.hatrhoFt = df.Function(self.CG1_sca)
        for idx in range(4, len(prod_terms)):
            if prod_terms[idx] is not None:
                self.hatrhoFdelta[idx-4] = df.Function(self.CG1_sca)
            else:
                self.hatrhoFdelta[idx-4] = df.Constant(0.0)

    def actualize_prod_terms(self):
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
                self.hatrhoFdelta[idx-4].assign(df.project(self.prod_terms[idx], self.CG1_sca))
            else:
                self.hatrhoFdelta[idx-4] = df.Constant(0.0)

    def metabolic_switch(self):
        u, p, nSh, nSt, nSn, cFt, cFdelta = self.unpack_prim_pvars(self.sol)
        H3 = df.conditional(df.gt(self.sol.sub(5), self.cFt_ms), 1.0, 0.0)
        cond = df.gt(nSt, 0.9875 * self.nSt_ms)
        cond = df.conditional(cond, nSt, H3 * self.nSt_ms)
        df.assign(self.sol.sub(3), df.project(cond, self.CG1_sca))


    def output(self, time) -> None:
        for idx, prim_var in enumerate(self.prim_vars_list):
            write_field2xdmf(self.output_file, self.sol.sub(idx), prim_var, time)
        #write_field2xdmf(self.output_file, df.project(nSt, self.CG1_sca, solver_type="cg"), "nSt", time)  # , self.eval_points, self.mesh)
        write_field2xdmf(self.output_file, self.intern_output[0], "nF", time, function_space=self.CG1_sca)  # , self.eval_points, self.mesh)
        write_field2xdmf(self.output_file, self.intern_output[1], "hatrhoS", time, function_space=self.CG1_sca)  # , self.eval_points, self.mesh)
        write_field2xdmf(self.output_file, self.intern_output[2], "hatrhoFt", time, function_space=self.CG1_sca)  # , self.eval_points, self.mesh)
        write_field2xdmf(self.output_file, self.intern_output[3], "hatrhoFn", time, function_space=self.CG1_sca)  # , self.eval_points, self.mesh)
        write_field2xdmf(self.output_file, self.intern_output[4], "DFt", time, function_space=self.CG1_sca)  # , self.eval_points, self.mesh)

    def unpack_prim_pvars(self, function_space: df.Function) -> tuple:
        """unpacks primary variables and returns tuple"""
        u = df.split(function_space)
        p = []
        for i in range(self.n_init_prim_vars, len(u)):
            p.append(u[i])
        return u[0], u[1], u[2], u[3], u[4], u[5], p 

    def set_weak_form(self) -> None:
        # Get Ansatz and test functions
        self.sol_old = df.Function(self.function_space)  # old primaries
        u, p, nSh, nSt, nSn, cFt, cFdelta = self.unpack_prim_pvars(self.ansatz_functions)
        _u, _p, _nSh, _nSt, _nSn, _cFt, _cFdelta = self.unpack_prim_pvars(self.test_functions)
        u_n, p_n, nSh_n, nSt_n, nSn_n, cFt_n, cFdelta_n = self.unpack_prim_pvars(self.sol_old)

        dx = df.Measure("dx", domain=self.mesh)

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

        # Calculate volume fractions
        nS = nSh + nSt + nSn
        rhoS = (nSh * df.Constant(self.rhoShR) + nSt * df.Constant(self.rhoStR) + nSn * df.Constant(self.rhoSnR)) / nS
        nF = 1.0 - nS

        ##############################################################################
        # Get growth terms
        #######################################
        hatnS = self.hatnSh + self.hatnSt + self.hatnSn
        hatrhoS = (self.hatnSh * df.Constant(self.rhoShR) + self.hatnSt * df.Constant(self.rhoStR) + self.hatnSn * df.Constant(self.rhoSnR)) / nS
        ##############################################################################
        # Time-dependent fields
        #######################################
        # Calculate velocity
        v = (u - u_n) / df.Constant(self.dt)
        div_v = ufl.inner(D_S, I)
        # Calculate storage terms
        dnShdt = (nSh - nSh_n) / df.Constant(self.dt)
        dnStdt = (nSt - nSt_n) / df.Constant(self.dt)
        dnSndt = (nSn - nSn_n) / df.Constant(self.dt)
        dcFtdt = (cFt - cFt_n) / df.Constant(self.dt)
        dcFdeltadt = []
        for i, cFd in enumerate(cFdelta):
            dcFdeltadt.append((cFd - cFdelta_n[i]) / df.Constant(self.dt))
        ##############################################################################

        ##############################################################################
        # Calculate Stress
        lambdaS = (df.Constant(self.lambdaSh) * nSh + df.Constant(self.lambdaSt) * nSt + df.Constant(self.lambdaSn) * nSn) / (nSh + nSt + nSn)
        muS = (df.Constant(self.muSh) * nSh + df.Constant(self.muSt) * nSt + df.Constant(self.muSn) * nSn) / (nSh + nSt + nSn)
        # Rodriguez Split
        B_Se = B_S
        J_Se = J_S
        if self.flag_defSplit:
            nS_n = nSh_n + nSt_n + nSn_n
            time = df.Constant(0)
            J_Sg = ufl.exp(hatnS / nS_n * time)
            F_Sg = df.dot(J_Sg ** (1 / len(u)), I)
            F_Se = F_S * ufl.inv(F_Sg)
            J_Se = ufl.det(F_Se)
            B_Se = F_Se * F_Se.T

        TS_E = (muS * (B_Se - I) + lambdaS * ufl.ln(J_Se) * I) / J_Se
        T = TS_E - p * I
        P = J_S * T * ufl.inv(F_S.T)
        ##############################################################################

        ##############################################################################
        # Define weak forms
        #######################################

        kD = df.Constant(self.kF) / df.Constant(self.gammaFR)
        dhrSdnF = hatrhoS / nF

        #######################################
        # Momentum balance of overall aggregate
        res_LMo1 = ufl.inner(P, ufl.grad(_u)) * dx
        res_LMo2 = - J_S * dhrSdnF * kD * ufl.inner(ufl.dot(ufl.grad(p), ufl.inv(F_S)) - dhrSdnF * v, _u) * dx
        res_LMo = res_LMo1 + res_LMo2
        #######################################

        #######################################
        # Volume balance of the mixture
        res_VBm1 = J_S * div_v * _p * dx
        res_VBm21 = ufl.dot(ufl.dot(ufl.grad(p), ufl.inv(F_S)) - dhrSdnF * v, ufl.inv(F_S.T))
        res_VBm2 = J_S * kD * ufl.inner(res_VBm21, ufl.grad(_p)) * dx
        res_VBm3 = - J_S * hatnS * (1.0 - rhoS / df.Constant(self.rhoFR)) * _p * dx  # ultra empfindlich
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
        nFcFtw_Ft1 = self.DFt * ufl.dot(ufl.grad(cFt), ufl.inv(C_S))
        nFcFtw_Ft2 = - kD * ufl.dot(ufl.dot(ufl.grad(p), ufl.inv(F_S)) - dhrSdnF * v, ufl.inv(F_S.T))
        nFcFtw_Ft = nFcFtw_Ft1 + nFcFtw_Ft2
        res_CBt1 = J_S * (nF * dcFtdt - self.hatrhoFt / df.Constant(self.molFt)) * _cFt * dx
        res_CBt2 = J_S * cFt * (div_v + hatrhoS / rhoS) * _cFt * dx
        res_CBt3 = cFt * ufl.inner(nFcFtw_Ft, ufl.grad(_cFt)) * dx
        res_CBt = res_CBt1 + res_CBt2 + res_CBt3
        #######################################

        #######################################
        # Concentration balance of additionals
        res_CBdelta = []
        for i, cFd in enumerate(cFdelta):
            nFcFdeltaw_Fdelta1 = self.DFdelta[i] * ufl.dot(ufl.grad(cFd), ufl.inv(C_S))
            nFcFdeltaw_Fdelta2 = - kD * ufl.dot(ufl.dot(ufl.grad(p), ufl.inv(F_S)) - dhrSdnF * v, ufl.inv(F_S.T))
            nFcFdeltaw_Fdelta = nFcFdeltaw_Fdelta1 + nFcFdeltaw_Fdelta2
            res_CBdelta1 = J_S * (nF * dcFdeltadt[i] - self.hatrhoFdelta[i] / df.Constant(self.molFdelta[i])) * _cFdelta[i] * dx
            res_CBdelta2 = J_S * cFd * (div_v + hatrhoS / rhoS) * _cFdelta[i] * dx
            res_CBdelta3 = J_S * cFd * ufl.inner(nFcFdeltaw_Fdelta, ufl.grad(_cFdelta[i])) * dx
            res_CBdelta.append(res_CBdelta1 + res_CBdelta2 + res_CBdelta3)
        #######################################

        # sum up to total residual
        res_tot = res_LMo + res_VBm + res_VBh + res_VBt + res_VBn + res_CBt
        for res_CBd in res_CBdelta:
            res_tot += res_CBd
        if self.n_bound is not None:
            res_tot += self.n_bound
        ##############################################################################
        self.intern_output = [nF, hatrhoS, self.hatrhoFt, self.hatrhoFdelta[0], self.DFt]
        self.residuum = res_tot

    def assign_if_function(self, var, index):
        if type(var) is df.Function:
            df.assign(self.sol.sub(index), var)
            df.assign(self.sol_old.sub(index), var)

    def set_initial_conditions(self, init, add):
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
            self.cFdelta_0S = add.cFdelta_0S
        # collect for interpolation
        init_set = gen.check_if_type(init.uS_0S, df.Function, None)
        init_set.append(gen.check_if_type(init.p_0S, df.Function, None))
        init_set.append(gen.check_if_type(init.nSh_0S, df.Function, None))
        init_set.append(gen.check_if_type(init.nSt_0S, df.Function, None))
        init_set.append(gen.check_if_type(init.nSn_0S, df.Function, None))
        init_set.append(gen.check_if_type(init.cFt_0S, df.Function, None))
        if self.cFdelta_0S is not None:
            for cFd_0S in self.cFdelta_0S:
                init_set.append(gen.check_if_type(cFd_0S, df.Function, None))
        self.sol.interpolate(InitialCondition(init_set))
        self.sol_old.interpolate(InitialCondition(init_set))

        self.assign_if_function(self.uS_0S, 0)
        self.assign_if_function(self.p_0S, 1)
        self.assign_if_function(self.nSh_0S, 2)
        self.assign_if_function(self.nSt_0S, 3)
        self.assign_if_function(self.nSn_0S, 4)
        self.assign_if_function(self.cFt_0S, 5)
        if self.cFdelta_0S is not None:
            for idx, cFd_0S in enumerate(self.cFdelta_0S):
                self.assign_if_function(cFd_0S, 5+idx)
        # production terms
        self.actualize_prod_terms()
        # metabolic switch
        self.metabolic_switch()

    def set_hets_if_needed(self, field):
        if type(field) is float:
            field = df.Constant(field)
        else:
            help_func = field
            field = df.Function(df.FunctionSpace(self.mesh, "DG", 0))
            field.interpolate(InitialDistribution(help_func))
        return field

    def set_heterogenities(self):
        self.kF = self.set_hets_if_needed(self.kF)
        self.lambdaSh = self.set_hets_if_needed(self.lambdaSh)
        self.lambdaSt = self.set_hets_if_needed(self.lambdaSt)
        self.lambdaSn = self.set_hets_if_needed(self.lambdaSn)
        self.muSh = self.set_hets_if_needed(self.muSh)
        self.muSt = self.set_hets_if_needed(self.muSt)
        self.muSn = self.set_hets_if_needed(self.muSn)
        self.DFt = self.set_hets_if_needed(self.DFt)
        for i in range(len(self.DFdelta)):
            self.DFdelta[i] = self.set_hets_if_needed(self.DFdelta[i])

    def set_solver(self):
        # Make sure quadrature_degree stays at 2
        prm = df.parameters["form_compiler"]
        prm["quadrature_degree"] = 2
        self.sol = self.ansatz_functions
        self.solver = solv.nonlinvarsolver(self.residuum, self.sol, self.d_bound, self.solver_param)

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
            # metabolic switch
            self.metabolic_switch()
            # Output solution
            if out_count >= self.output_interval:  #-self.dt:
                print("Time: {}".format(t), "  ", "Converged in steps: {}".format(n_iter))
                out_count = 0.0
                self.output(t)
            # Update history fields
            self.sol_old.assign(self.sol)
