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
from abc import ABC
import oncofem.modelling.base_model.solver as solv
from oncofem import Problem
from oncofem.helper.io import write_field2xdmf
import dolfin as df
import ufl

#############################################################
#                                                           #
#  Helper functions                                         #
#                                                           #
#############################################################
class InitialDistribution(df.UserExpression):
    def __init__(self, value, **kwargs):
        self.value = value
        super().__init__(**kwargs)
    def eval_cell(self, values, x, cell):
        values[0] = self.value

class InitialCondition(df.UserExpression, ABC):
    def __init__(self, init_set,  **kwargs):
        self.init_set = self.case_distinction(init_set)
        self.size = len(init_set)
        super().__init__(**kwargs)

    def case_distinction(self, init_cond: list):
        for idx,cond in enumerate(init_cond):
            if cond is None:
                init_cond[idx] = None
            elif type(cond) is float or type(cond) is df.UserExpression:
                init_cond[idx] = cond
            else:
                print("unhandled exception!")
        return init_cond

    def eval_cell(self, values, x, cell):
        for idx,val in enumerate(values):
            if type(self.init_set[idx]) is float:
                values[idx] = self.init_set[idx]
            else:
                values[idx] = self.init_set[idx][cell.index]

    def value_shape(self):
        return self.size,


class InitialConditionInternals(df.UserExpression):  # UserExpression instead of Expression
    def __init__(self, subdomains, **kwargs):
        super().__init__(**kwargs)
        self.subdomains = subdomains

    def eval_cell(self, values, x, cell):
        values[0] = 0.0  # hatnS
        if self.subdomains[cell.index] == 5:
            values[0] = 1e-2  # hatnS

#############################################################
#                                                           #
#  Helper functions                                         #
#                                                           #
#############################################################
class Glioblastoma:

    def __init__(self):
        # general infos
        self.output_file = None
        self.flag_proliferation = True
        self.flag_metabolism = True
        self.flag_apoptose = True
        self.flag_necrosis = True
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
        self.DG1 = None
        self.V0 = None
        self.V1 = None
        self.V2 = None
        self.sol = None
        self.sol_old = None

        # geometry paramereters
        self.mesh = None
        self.dim = None
        self.n_bound = None
        self.d_bound = None

        # 
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

        # spatial varying material parameters
        self.nSh_0S = None
        self.nSt_0S = None
        self.nSn_0S = None
        self.nF_0S = None
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

    def set_param(self, input: Problem):
        """
        sets parameter needed for model class
        """
        # general parameters
        self.output_file = input.param.gen.output_file
        self.flag_proliferation = input.param.gen.flag_proliferation
        self.flag_metabolism = input.param.gen.flag_metabolism
        self.flag_apoptose = input.param.gen.flag_apop
        self.flag_necrosis = input.param.gen.flag_necrosis
        self.flag_defSplit = input.param.gen.flag_defSplit

        # time parameters
        self.T_end = input.param.time.T_end
        self.output_interval = input.param.time.output_interval
        self.dt = input.param.time.dt

        # material parameters base model
        self.rhoShR = df.Constant(input.param.mat.rhoShR)
        self.rhoStR = df.Constant(input.param.mat.rhoStR)
        self.rhoSnR = df.Constant(input.param.mat.rhoSnR)
        self.rhoFR = df.Constant(input.param.mat.rhoFR)
        self.gammaFR = df.Constant(input.param.mat.gammaFR)
        self.molFt = df.Constant(input.param.mat.molFt)

        # spatial varying material parameters
        self.kF = input.param.mat.kF
        self.lambdaSh = input.param.mat.lambdaSh
        self.lambdaSt = input.param.mat.lambdaSt
        self.lambdaSn = input.param.mat.lambdaSn
        self.muSh = input.param.mat.muSh
        self.muSt = input.param.mat.muSt
        self.muSn = input.param.mat.muSn
        self.DFt = input.param.mat.DFt

        # initial conditions
        self.uS_0S = input.param.init.uS_0S
        self.p_0S = input.param.init.p_0S 
        self.nSh_0S = input.param.init.nSh_0S
        self.nSt_0S = input.param.init.nSt_0S
        self.nSn_0S = input.param.init.nSn_0S
        self.nF_0S = input.param.init.nF_0S
        self.cFt_0S = input.param.init.cFt_0S

        # FEM paramereters and additionals
        self.solver_param = input.param.fem.solver_param
        if hasattr(input.param.add, "prim_vars"):
            self.prim_vars_list.extend(input.param.add.prim_vars)
            self.ele_types.extend(input.param.add.ele_types)
            self.ele_orders.extend(input.param.add.ele_orders)
            self.tensor_order.extend(input.param.add.tensor_orders)
            self.molFdelta = input.param.add.molFdelta
            self.DFdelta = input.param.add.DFdelta
            self.cFdelta_0S = input.param.add.cFdelta_0S

        # geometry parameters
        self.mesh = input.geom.mesh
        self.dim = input.geom.dim
        self.n_bound = input.geom.n_bound
        self.d_bound = input.geom.d_bound

    def set_function_spaces(self):
        """
            sets function space for primary variables u, p, cIn, cIt, cIv and for internal variables
        """
        elements = []
        for idx, type in enumerate(self.ele_types):
            if self.tensor_order[idx] == 0:
                elements.append(df.FiniteElement(type, self.mesh.ufl_cell(), self.ele_orders[idx]))
            elif self.tensor_order[idx] == 1:
                elements.append(df.VectorElement(type, self.mesh.ufl_cell(), self.ele_orders[idx]))
            elif self.tensor_order[idx] == 2:
                elements.append(df.TensorElement(type, self.mesh.ufl_cell(), self.ele_orders[idx]))
        self.finite_element = df.MixedElement(elements)
        self.function_space = df.FunctionSpace(self.mesh, self.finite_element)
        self.DG0 = df.FunctionSpace(self.mesh, "DG", 0)
        self.DG1 = df.FunctionSpace(self.mesh, "DG", 1)
        self.V0 = df.FunctionSpace(self.mesh, "P", 1)
        self.V1 = df.VectorFunctionSpace(self.mesh, "P", 1)
        self.V2 = df.TensorFunctionSpace(self.mesh, "P", 1)
        self.ansatz_functions = df.Function(self.function_space)
        self.test_functions = df.TestFunction(self.function_space)

    def set_bio_chem_models(self, input):
        self.bm_model_prolif_cFt    = input.bmm.bm_model_prolif_cFt
        self.bm_model_prolif_nSt    = input.bmm.bm_model_prolif_nSt
        self.bm_model_necros_nSh    = input.bmm.bm_model_necros_nSh
        self.bm_model_necros_nSt    = input.bmm.bm_model_necros_nSt
        self.bm_model_necros_cFt    = input.bmm.bm_model_necros_cFt
        self.bm_model_apopto_nSh    = input.bmm.bm_model_apopto_nSh
        self.bm_model_apopto_nSt    = input.bmm.bm_model_apopto_nSt
        self.bm_model_apopto_cFt    = input.bmm.bm_model_apopto_cFt
        self.bm_model_apopto_cFa    = input.bmm.bm_model_apopto_cFa
        self.bm_model_metabo_cFn    = input.bmm.bm_model_metabo_cFn

    def output(self, time):
        for idx, prim_var in enumerate(self.prim_vars_list):
            write_field2xdmf(self.output_file, self.sol.sub(idx), prim_var, time)
        #write_field2xdmf(self.output_file, df.project(nSt, self.V0, solver_type="cg"), "nSt", time)  # , self.eval_points, self.mesh)
        write_field2xdmf(self.output_file, df.project(self.intern_output[0], self.V0, solver_type="cg"), "nF", time)  # , self.eval_points, self.mesh)
        write_field2xdmf(self.output_file, df.project(self.intern_output[1], self.V0, solver_type="cg"), "hatrhoS", time)  # , self.eval_points, self.mesh)

    def unpack_prim_pvars(self, function_space: df.Function):
        u = df.split(function_space)
        p = []
        for i in range(self.n_init_prim_vars, len(u)):
            p.append(u[i])
        return u[0], u[1], u[2], u[3], u[4], u[5], p 

    def set_weak_form(self):
        # Get Ansatz and test functions
        self.sol_old = df.Function(self.function_space)  # old primaries
        u, p, nSh, nSt, nSn, cFt, cFdelta = self.unpack_prim_pvars(self.ansatz_functions)
        _u, _p, _nSh, _nSt, _nSn, _cFt, _cFdelta = self.unpack_prim_pvars(self.test_functions)
        u_n, p_n, nSh_n, nSt_n, nSn_n, cFt_n, cFdelta_n = self.unpack_prim_pvars(self.sol_old)
        dx = df.dx

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
        # Calculate growth terms
        #######################################
        hatnS = df.Constant(0.0)
        hatrhoS = df.Constant(0.0)
        hatnSh = df.Constant(0.0)
        hatnSt = df.Constant(0.0)
        hatnSn = df.Constant(0.0)
        hatrhoFt = df.Constant(0.0)
        hatrhoFdelta = []
        for i in range(len(cFdelta)):
            hatrhoFdelta[i] = df.Constant(0.0)
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
            F_Sg = J_Sg ** (1 / len(u)) * I
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
        res_VBh1 = J_S * (dnShdt - hatnSh) * _nSh * dx
        res_VBh2 = J_S * nSh * div_v * _nSh * dx
        res_VBh = res_VBh1 + res_VBh2
        #######################################

        #######################################
        # Volume balance of tumor cells
        res_VBt1 = J_S * (dnStdt - hatnSt) * _nSt * dx
        res_VBt2 = J_S * nSt * div_v * _nSt * dx
        res_VBt = res_VBt1 + res_VBt2
        #######################################

        #######################################
        # Volume balance of necrotic cells
        res_VBn1 = J_S * (dnSndt - hatnSn) * _nSn * dx
        res_VBn2 = J_S * nSn * div_v * _nSn * dx
        res_VBn = res_VBn1 + res_VBn2
        #######################################

        #######################################
        # Concentration balance of solved cancer cells
        nFcFtw_Ft1 = + df.Constant(self.DFt) * ufl.dot(ufl.grad(cFt), ufl.inv(C_S))
        nFcFtw_Ft2 = - kD * ufl.dot(ufl.dot(ufl.grad(p), ufl.inv(F_S)) - dhrSdnF * v, ufl.inv(F_S.T))
        nFcFtw_Ft = nFcFtw_Ft1 + nFcFtw_Ft2
        res_CBt1 = J_S * (nF * dcFtdt - hatrhoFt / df.Constant(self.molFt)) * _cFt * dx
        res_CBt2 = J_S * cFt * (div_v + hatrhoS / rhoS) * _cFt * dx
        res_CBt3 = cFt * ufl.inner(nFcFtw_Ft, ufl.grad(_cFt)) * dx
        res_CBt = res_CBt1 + res_CBt2 + res_CBt3
        #######################################

        #######################################
        # Concentration balance of additionals
        res_CBdelta = []
        for i, cFd in enumerate(cFdelta):
            nFcFdeltaw_Fdelta1 = + df.Constant(self.DFdelta[i]) * ufl.dot(ufl.grad(cFd), ufl.inv(C_S))
            nFcFdeltaw_Fdelta2 = - kD * ufl.dot(ufl.dot(ufl.grad(p), ufl.inv(F_S)) - dhrSdnF * v, ufl.inv(F_S.T))
            nFcFdeltaw_Fdelta = nFcFdeltaw_Fdelta1 + nFcFdeltaw_Fdelta2
            res_CBdelta1 = J_S * (nF * dcFdeltadt[i] - hatrhoFdelta[i] / df.Constant(self.molFdelta[i])) * _cFdelta[i] * dx
            res_CBdelta2 = J_S * cFd * (div_v + hatrhoS / rhoS) * _cFdelta[i] * dx
            res_CBdelta3 = J_S * cFd * ufl.inner(nFcFdeltaw_Fdelta, ufl.grad(_cFdelta[i])) * dx
            res_CBdelta.append(res_CBdelta1 + res_CBdelta2 + res_CBdelta3)
        #######################################

        # sum up to total residual
        res_tot = res_LMo + res_VBm + res_VBh + res_VBt + res_VBn + res_CBt
        for res_CBd in res_CBdelta:
            res_tot += res_CBd
        if not self.n_bound is None:
            res_tot += self.n_bound
        ##############################################################################
        self.intern_output = [nF, hatrhoS]
        self.residuum = res_tot

    def set_initial_conditions(self):
        init_set = self.uS_0S
        init_set.append(self.p_0S)
        init_set.append(self.nSh_0S)
        init_set.append(self.nSt_0S)
        init_set.append(self.nSn_0S)
        init_set.append(self.cFt_0S)
        if self.cFdelta_0S is not None:
            for cFd_0S in self.cFdelta_0S:
                init_set.append(cFd_0S)

        self.sol.interpolate(InitialCondition(init_set))
        self.sol_old.interpolate(InitialCondition(init_set))

    def set_heterogenities(self):
        # TODO: Eigentlich müssen die Größen als input parameter auf die felder geschoben werden. demnach müssen in der Schwachen form eigentlich auch zunächst Felder initialisiert werden.
        #self.kF = input.param.mat.kF
        #self.lambdaSh = input.param.mat.lambdaSh
        #self.lambdaSt = input.param.mat.lambdaSt
        #self.lambdaSn = input.param.mat.lambdaSn
        #self.muSh = input.param.mat.muSh
        #self.muSt = input.param.mat.muSt
        #self.muSn = input.param.mat.muSn
        #self.DFt = input.param.mat.DFt
        #DFn.interpolate(self.DFn_distr)
        #DFt.interpolate(self.DFt_distr)
        #DFa.interpolate(self.DFa_distr)
        pass

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
            # Output solution
            if out_count >= self.output_interval:  #-self.dt:
                print("Time: {}".format(t), "  ", "Converged in steps: {}".format(n_iter))
                out_count = 0.0
                self.output(t)
            # Update history fields
            self.sol_old.assign(self.sol)
