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
import dolfin.cpp.mesh
import oncofem.modelling.base_model.solver as solv
from oncofem.helper.io import write_field2xdmf
import dolfin as df
import ufl

#############################################################
#                                                           #
#  Helper functions                                         #
#                                                           #
#############################################################
class InitialCondition(df.UserExpression):  # UserExpression instead of Expression
    def __init__(self, edema_distr, active_tumor_distr, necrotic_distr,  **kwargs):
        super().__init__(**kwargs)
        self.edema_distr = edema_distr
        self.active_tumor_distr = active_tumor_distr
        self.necrotic_distr = necrotic_distr

    def eval_cell(self, values, x, cell):
        values[0] = 0.0  # u_x
        values[1] = 0.0  # u_y
        values[2] = 0.0  # u_z
        values[3] = 0.0  # p
        values[4] = 0.8 #- self.active_tumor_distr[cell.index] - self.necrotic_distr[cell.index]   # nSh #TODO: Fix Value
        values[5] = 0#self.active_tumor_distr[cell.index]  # nSt
        values[6] = 0#self.necrotic_distr[cell.index]  # nSn
        values[7] = 0.0  # cFn
        values[8] = 0.0001 #self.edema_distr[cell.index]  # cFt
        values[9] = 0.0  # cFv
        values[10] = 0.0  # cFa

    def value_shape(self):
        return 11,

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
        # General infos
        self.output_file = None
        self.flag_proliferation = True
        self.flag_metabolism = True
        self.flag_apoptose = True
        self.flag_necrosis = True
        self.flag_defSplit = True

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

        self.prim_vars_list = None
        self.tensor_order = None
        self.ele_types = None
        self.ele_orders = None

        self.mesh = None
        self.domain = None
        self.edema_distr = None
        self.active_tumor_distr = None
        self.necrotic_distr = None
        self.growthArea = None

        self.dx = None
        self.residuum = None
        self.n_bound = None
        self.d_bound = None
        self.initial_condition = None
        self.internal_condition = None

        # Material Parameters
        self.rhoShR = None
        self.rhoStR = None
        self.rhoSnR = None
        self.rhoFR = None
        self.gammaFR = None
        self.molFn = None
        self.molFt = None
        self.molFv = None
        self.molFa = None
        self.kF = None
        self.DFn_distr = None
        self.DFt_distr = None
        self.DFv_distr = None
        self.DFa_distr = None
        self.lambdaSh = None
        self.lambdaSt = None
        self.lambdaSn = None
        self.muSh = None
        self.muSt = None
        self.muSn = None

        # Time Parameters
        self.time = None
        self.T_end = None
        self.dt = None

        # FEM Paramereters
        self.solver_param = None

    def set_heterogenities(self):
        self.initial_condition = InitialCondition(self.edema_distr, self.active_tumor_distr, self.necrotic_distr)

    def set_boundaries(self, d_bound, n_bound):
        self.d_bound = d_bound
        self.n_bound = n_bound

    def set_param(self, input):
        """
        sets parameter needed for model class
        """
        self.output_file = input.param.gen.output_file
        self.flag_proliferation = input.param.gen.flag_proliferation
        self.flag_metabolism = input.param.gen.flag_metabolism
        self.flag_apoptose = input.param.gen.flag_apop
        self.flag_necrosis = input.param.gen.flag_necrosis
        self.flag_defSplit = input.param.gen.flag_defSplit

        self.prim_vars_list = input.param.fem.prim_vars
        self.tensor_order = input.param.fem.tensor_order
        self.ele_types = input.param.fem.ele_types
        self.ele_orders = input.param.fem.ele_orders

        self.mesh = input.geom.mesh
        self.domain = input.geom.domain
        #self.edema_distr = input.geom.edema_distr
        #self.solid_tumor_distr = input.geom.solid_tumor_distr
        #self.necrotic_distr = input.geom.necrotic_distr
        self.dx = input.geom.dx
        self.n_bound = input.geom.n_bound
        self.d_bound = input.geom.d_bound
        self.rhoShR = df.Constant(input.param.mat.rhoShR)
        self.rhoStR = df.Constant(input.param.mat.rhoStR)
        self.rhoSnR = df.Constant(input.param.mat.rhoSnR)
        self.rhoFR = df.Constant(input.param.mat.rhoFR)
        self.gammaFR = df.Constant(input.param.mat.gammaFR)
        self.molFn = df.Constant(input.param.mat.molFn)
        self.molFt = df.Constant(input.param.mat.molFt)
        self.molFv = df.Constant(input.param.mat.molFv)
        self.molFa = df.Constant(input.param.mat.molFa)
        self.kF = df.Constant(input.param.mat.kF)
        #self.DFn_distr = input.param.mat.DFn_distr
        #self.DFt_distr = input.param.mat.DFt_distr
        #self.DFa_distr = input.param.mat.DFa_distr
        self.lambdaSh = df.Constant(input.param.mat.lambdaSh)
        self.lambdaSt = df.Constant(input.param.mat.lambdaSt)
        self.lambdaSn = df.Constant(input.param.mat.lambdaSn)
        self.muSh = df.Constant(input.param.mat.muSh)
        self.muSt = df.Constant(input.param.mat.muSt)
        self.muSn = df.Constant(input.param.mat.muSn)
        self.T_end = input.param.time.T_end
        self.dt = input.param.time.dt
        self.solver_param = input.param.fem.solver_param

    def set_function_spaces(self):
        """
            sets function space for primary variables u, p, cIn, cIt, cIv and for internal variables
        """
        elements = []
        for idx, type in enumerate(self.ele_types):
            if self.tensor_order[idx] == 0:
                elements.append(df.FiniteElement(type, self.mesh.ufl_cell(), self.ele_orders[idx]))
            if self.tensor_order[idx] == 1:
                elements.append(df.VectorElement(type, self.mesh.ufl_cell(), self.ele_orders[idx]))
            if self.tensor_order[idx] == 2:
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

    def set_weak_form(self):
        # Material Parameters and Parameters that go into weak form
        rhoShR = df.Constant(self.rhoShR)
        rhoStR = df.Constant(self.rhoStR)
        rhoSnR = df.Constant(self.rhoSnR)
        rhoFR = df.Constant(self.rhoFR)
        gammaFR = df.Constant(self.gammaFR)
        molFn = df.Constant(self.molFn)
        molFt = df.Constant(self.molFt)
        molFv = df.Constant(self.molFv)
        molFa = df.Constant(self.molFa)
        kF = df.Constant(self.kF)
        DFn = df.Function(self.DG0)
        DFt = df.Function(self.DG0)  # .interpolate(self.DFt_distr)
        DFv = df.Function(self.DG0)  # .interpolate(self.DFt_distr)
        DFa = df.Function(self.DG0)  # .interpolate(self.DFa_distr)
        lambdaSh = df.Constant(self.lambdaSh)
        lambdaSt = df.Constant(self.lambdaSt)
        lambdaSn = df.Constant(self.lambdaSn)
        muSh = df.Constant(self.muSh)
        muSt = df.Constant(self.muSt)
        muSn = df.Constant(self.muSn)
        dt = df.Constant(self.dt)

        # Get Ansatz and test functions
        self.sol_old = df.Function(self.function_space)  # old primaries
        u, p, nSh, nSt, nSn, cFn, cFt, cFv, cFa = df.split(self.ansatz_functions)
        _u, _p, _nSh, _nSt, _nSn, _cFn, _cFt, _cFv, _cFa = df.split(self.test_functions)
        u_n, p_n, nSh_n, nSt_n, nSn_n, cFn_n, cFt_n, cFv_n, cFa_n = df.split(self.sol_old)

        dx = self.dx

        # Kinematics
        I = ufl.Identity(len(u))
        F_S = I + ufl.grad(u)
        F_Sn = I + ufl.grad(u_n)
        J_S = ufl.det(F_S)
        C_S = F_S.T * F_S
        B_S = F_S * F_S.T
        dF_Sdt = (F_S - F_Sn) / dt
        L_S = dF_Sdt * ufl.inv(F_S)
        D_S = (L_S + L_S.T) / 2.0

        # Calculate volume fractions
        nS = nSh + nSt + nSn
        rhoS = (nSh * rhoShR + nSt * rhoStR + nSn * rhoSnR) / nS
        nF = 1.0 - nS

        ##############################################################################
        # Calculate growth terms
        #######################################
        # Define processes

        # Proliferation
        hatrhoFt_prolif = df.Constant(0.0)
        hatrhoSt_prolif = df.Constant(0.0)
        if self.flag_proliferation:
            hatrhoFt_prolif = self.bm_model_prolif_cFt
            hatrhoSt_prolif = self.bm_model_prolif_nSt

        # Necrosis
        hatrhoSh_necros = df.Constant(0.0)
        hatrhoSt_necros = df.Constant(0.0)
        hatrhoSn_necros = df.Constant(0.0)
        hatrhoFt_necros = df.Constant(0.0)
        if self.flag_necrosis:
            hatrhoSh_necros = self.bm_model_necros_nSh
            hatrhoSt_necros = self.bm_model_necros_nSt
            hatrhoSn_necros = - (hatrhoSt_necros + hatrhoSh_necros)
            hatrhoFt_necros = self.bm_model_necros_cFt

        # Apoptose
        hatrhoSh_apop = df.Constant(0.0)
        hatrhoSt_apop = df.Constant(0.0)
        hatrhoFt_apop = df.Constant(0.0)
        hatrhoFa_apop = df.Constant(0.0)
        if self.flag_apoptose:
            hatrhoSh_apop = self.bm_model_apopto_nSh
            hatrhoSt_apop = self.bm_model_apopto_nSt
            hatrhoFt_apop = self.bm_model_apopto_cFt
            hatrhoFa_apop = self.bm_model_apopto_cFa

        # Metabolism
        hatrhoFn = df.Constant(0.0)
        if self.flag_metabolism:
            hatrhoFn = self.bm_model_metabo_cFn
        #######################################
        # Accumulation
        hatrhoSh = hatrhoSh_apop + hatrhoSh_necros
        hatrhoSt = hatrhoSt_apop + hatrhoSt_necros + hatrhoSt_prolif
        hatrhoSn = hatrhoSn_necros
        hatrhoFt = hatrhoFt_apop + hatrhoFt_necros + hatrhoFt_prolif
        hatrhoFv = df.Constant(0.0)
        hatrhoFa = hatrhoFa_apop

        #######################################
        # express growth via different quantities
        hatrhoS = hatrhoSt + hatrhoSn + hatrhoSh
        hatnS = hatrhoS / rhoS
        hatnSh = hatrhoSh / rhoStR
        hatnSt = hatrhoSt / rhoStR
        hatnSn = hatrhoSn / rhoStR
        ##############################################################################

        ##############################################################################
        # Time-dependent fields
        #######################################
        # Calculate velocity
        v = (u - u_n) / dt
        div_v = ufl.inner(D_S, I)
        # Calculate storage terms
        dnShdt = (nSh - nSh_n) / dt
        dnStdt = (nSt - nSt_n) / dt
        dnSndt = (nSn - nSn_n) / dt
        dcFndt = (cFn - cFn_n) / dt
        dcFtdt = (cFt - cFt_n) / dt
        dcFvdt = (cFv - cFv_n) / dt
        dcFadt = (cFa - cFa_n) / dt
        ##############################################################################

        ##############################################################################
        # Calculate Stress
        lambdaS = (lambdaSh * nSh + lambdaSt * nSt + lambdaSn * nSn) / (nSh + nSt + nSn)
        muS = (muSh * nSh + muSt * nSt + muSn * nSn) / (nSh + nSt + nSn)
        # Rodriguez Split
        B_Se = B_S
        J_Se = J_S
        if self.flag_defSplit == True:
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

        kD = kF / gammaFR
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
        res_VBm3 = - J_S * hatnS * (1.0 - rhoS / rhoFR) * _p * dx  # ultra empfindlich
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
        # Concentration balance of solved nutrients
        nFcFnw_Fn1 = - DFn * ufl.dot(ufl.grad(cFn), ufl.inv(C_S))
        nFcFnw_Fn2 = - kD * ufl.dot(ufl.dot(ufl.grad(p), ufl.inv(F_S)) - dhrSdnF * v, ufl.inv(F_S.T))
        nFcFnw_Fn = nFcFnw_Fn1 + nFcFnw_Fn2
        res_CBn1 = J_S * (nF * dcFndt - hatrhoFn / molFn) * _cFn * dx
        res_CBn2 = J_S * cFn * (div_v + hatrhoS / rhoS) * _cFn * dx
        res_CBn3 = J_S * cFn * ufl.inner(nFcFnw_Fn, ufl.grad(_cFn)) * dx
        res_CBn = res_CBn1 + res_CBn2 + res_CBn3
        #######################################

        #######################################
        # Concentration balance of solved cancer cells
        nFcFtw_Ft1 = - DFt * ufl.dot(ufl.grad(cFt), ufl.inv(C_S))
        nFcFtw_Ft2 = - kD * ufl.dot(ufl.dot(ufl.grad(p), ufl.inv(F_S)) - dhrSdnF * v, ufl.inv(F_S.T))
        nFcFtw_Ft = nFcFtw_Ft1 + nFcFtw_Ft2
        res_CBt1 = J_S * (nF * dcFtdt - hatrhoFt / molFt) * _cFt * dx
        res_CBt2 = J_S * cFt * (div_v + hatrhoS / rhoS) * _cFt * dx
        res_CBt3 = J_S * cFt * ufl.inner(nFcFtw_Ft, ufl.grad(_cFt)) * dx
        res_CBt = res_CBt1 + res_CBt2 + res_CBt3
        #######################################

        #######################################
        # Concentration balance of solved cancer cells
        nFcFvw_Ft1 = - DFv * ufl.dot(ufl.grad(cFv), ufl.inv(C_S))
        nFcFvw_Ft2 = - kD * ufl.dot(ufl.dot(ufl.grad(p), ufl.inv(F_S)) - dhrSdnF * v, ufl.inv(F_S.T))
        nFcFvw_Ft = nFcFvw_Ft1 + nFcFvw_Ft2
        res_CBv1 = J_S * (nF * dcFvdt - hatrhoFv / molFv) * _cFv * dx
        res_CBv2 = J_S * cFv * (div_v + hatrhoS / rhoS) * _cFv * dx
        res_CBv3 = J_S * cFv * ufl.inner(nFcFvw_Ft, ufl.grad(_cFv)) * dx
        res_CBv = res_CBv1 + res_CBv2 + res_CBv3
        #######################################

        #######################################
        # Concentration balance of solved Drug
        nFcFaw_Fa1 = - DFa * ufl.dot(ufl.grad(cFa), ufl.inv(C_S))
        nFcFaw_Fa2 = - kD * ufl.dot(ufl.dot(ufl.grad(p), ufl.inv(F_S)) - dhrSdnF * v, ufl.inv(F_S.T))
        nFcFaw_Fa = nFcFaw_Fa1 + nFcFaw_Fa2
        res_CBa1 = J_S * (nF * dcFadt - hatrhoFa / molFa) * _cFa * dx
        res_CBa2 = J_S * cFa * (div_v + hatrhoS / rhoS) * _cFa * dx
        res_CBa3 = J_S * ufl.inner(nFcFaw_Fa, ufl.grad(_cFa)) * dx
        res_CBa = res_CBa1 + res_CBa2 + res_CBa3
        #######################################
        # sum up to total residual
        res_tot = res_LMo + res_VBm + res_VBh + res_VBt + res_VBn + res_CBn + res_CBt + res_CBv + res_CBa
        if not self.n_bound is None:
            res_tot += self.n_bound
        ##############################################################################
        self.residuum = res_tot

    def set_initial_conditions(self):
        self.sol_old.interpolate(self.initial_condition)
        self.sol.interpolate(self.initial_condition)
        # DFn.interpolate(self.DFn_distr)
        # DFt.interpolate(self.DFt_distr)
        # DFa.interpolate(self.DFa_distr)

    def solve(self):
        # Define problem solution
        self.sol = self.ansatz_functions
        solver = solv.nonlinvarsolver(self.residuum, self.sol, self.d_bound, self.solver_param)
        # Make sure quadrature_degree stays at 2
        prm = df.parameters["form_compiler"]
        prm["quadrature_degree"] = 2
        # Set initial conditions
        self.set_initial_conditions()
        # Initialize  and time loop
        t = 0
        self.output(t)
        print("Initial step is written")
        while t < self.T_end:
            # Increment solution time
            t = t + self.dt
            # Calculate current solution
            n_iter, converged = solver.solve()
            print("Time: {}".format(t), "  ", "Converged in steps: {}".format(n_iter))
            # Output solution
            self.output(t)
            # Update history fields
            self.sol_old.assign(self.sol)
