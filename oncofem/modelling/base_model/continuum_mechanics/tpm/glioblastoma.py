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
from oncofem.modelling.base_model.continuum_mechanics.constitutives import calcStress_vonMises
from oncofem.helper.io import write_field2output
import dolfin as df
import ufl

#############################################################
#                                                           #
#  Helper functions                                         #
#                                                           #
#############################################################
class InitialCondition(df.UserExpression):  # UserExpression instead of Expression
    def __init__(self, subdomains, **kwargs):
        super().__init__(**kwargs)
        self.subdomains = subdomains

    def eval_cell(self, values, x, cell):
        values[0] = 0.0  # u_x
        values[1] = 0.0  # u_y
        values[2] = 0.0  # u_z
        values[3] = 0.0  # p
        values[4] = 0.5  # nSh
        values[5] = 0.0  # nSt
        values[6] = 0.0  # nSn
        values[7] = 0.0  # cFn
        values[8] = 0.0  # cFt
        values[9] = 0.0  # cFv
        values[10] = 0.0  # cFa
        if (x[0] - 117.0) * (x[0] - 117.0) + (x[1] - 123.0) * (x[1] - 123.0) + (x[2] - 76.0) * (x[2] - 76.0) <= 100:
            values[0] = 0.0  # u_x
            values[1] = 0.0  # u_y
            values[2] = 0.0  # u_z
            values[3] = 0.0  # p
            values[4] = 0.5  # nSh
            values[5] = 0.0  # nSt
            values[6] = 0.0  # nSn
            values[7] = 1.0  # cFn
            values[8] = 0.0  # cFt
            values[9] = 0.0  # cFv
            values[10] = 0.0  # cFa
        if self.subdomains[cell.index] == 5:
            values[0] = 0.0  # u_x
            values[1] = 0.0  # u_y
            values[2] = 0.0  # u_z
            values[3] = 0.0  # p
            values[4] = 0.5  # nSh
            values[5] = 0.0  # nSt
            values[6] = 0.0  # nSn
            values[7] = 0.0  # cFn
            values[8] = 0.5e-2  # cFt
            values[9] = 0.0  # cFv
            values[10] = 0.0  # cFa

    def value_shape(self):
        return (11,)


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
        self.flag_angiogenesis = True
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

        self.type_u = None
        self.type_p = None
        self.type_nSh = None
        self.type_nSt = None
        self.type_nSn = None
        self.type_cFn = None
        self.type_cFt = None
        self.type_cFv = None
        self.type_cFa = None
        self.order_u = None
        self.order_p = None
        self.order_nSh = None
        self.order_nSt = None
        self.order_nSn = None
        self.order_cFn = None
        self.order_cFt = None
        self.order_cFv = None
        self.order_cFa = None
        self.mesh = None
        self.domain = None
        self.growthArea = None

        self.dx = None
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
        self.DFn = None
        self.DFt = None
        self.DFv = None
        self.DFa = None
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

    def set_initial_condition(self):
        self.initial_condition = InitialCondition(self.domain)
        self.internal_condition = InitialConditionInternals(self.domain)

    def set_boundaries(self, d_bound, n_bound):
        self.d_bound = d_bound
        self.n_bound = n_bound

    def set_param(self, input):
        """
        sets parameter needed for model class
        """
        self.output_file = input.param.gen.output_file
        self.flag_angiogenesis = input.param.gen.flag_angiogenesis
        self.flag_proliferation = input.param.gen.flag_proliferation
        self.flag_metabolism = input.param.gen.flag_metabolism
        self.flag_apoptose = input.param.gen.flag_apop
        self.flag_necrosis = input.param.gen.flag_necrosis
        self.flag_defSplit = input.param.gen.flag_defSplit
        self.type_u = input.param.fem.type_u
        self.type_p = input.param.fem.type_p
        self.type_nSh = input.param.fem.type_nSh
        self.type_nSt = input.param.fem.type_nSt
        self.type_nSn = input.param.fem.type_nSn
        self.type_cFn = input.param.fem.type_cFn
        self.type_cFt = input.param.fem.type_cFt
        self.type_cFv = input.param.fem.type_cFv
        self.type_cFa = input.param.fem.type_cFa
        self.order_u = input.param.fem.order_u
        self.order_p = input.param.fem.order_p
        self.order_nSh = input.param.fem.order_nSh
        self.order_nSt = input.param.fem.order_nSt
        self.order_nSn = input.param.fem.order_nSn
        self.order_cFn = input.param.fem.order_cFn
        self.order_cFt = input.param.fem.order_cFt
        self.order_cFv = input.param.fem.order_cFv
        self.order_cFa = input.param.fem.order_cFa
        self.mesh = input.geom.mesh
        self.domain = input.geom.domain
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
        self.DFn = df.Constant(input.param.mat.DFn)
        self.DFt = df.Constant(input.param.mat.DFt)
        self.DFv = df.Constant(input.param.mat.DFv)
        self.DFa = df.Constant(input.param.mat.DFa)
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
        e_u = df.VectorElement(self.type_u, self.mesh.ufl_cell(), self.order_u)
        e_p = df.FiniteElement(self.type_p, self.mesh.ufl_cell(), self.order_p)
        e_nSh = df.FiniteElement(self.type_nSh, self.mesh.ufl_cell(), self.order_nSh)
        e_nSt = df.FiniteElement(self.type_nSt, self.mesh.ufl_cell(), self.order_nSt)
        e_nSn = df.FiniteElement(self.type_nSn, self.mesh.ufl_cell(), self.order_nSn)
        e_cFn = df.FiniteElement(self.type_cFn, self.mesh.ufl_cell(), self.order_cFn)
        e_cFt = df.FiniteElement(self.type_cFt, self.mesh.ufl_cell(), self.order_cFt)
        e_cFv = df.FiniteElement(self.type_cFv, self.mesh.ufl_cell(), self.order_cFv)
        e_cFa = df.FiniteElement(self.type_cFa, self.mesh.ufl_cell(), self.order_cFa)
        self.finite_element = df.MixedElement(e_u, e_p, e_nSh, e_nSt, e_nSn, e_cFn, e_cFt, e_cFv, e_cFa)
        self.function_space = df.FunctionSpace(self.mesh, self.finite_element)
        self.DG0 = df.FunctionSpace(self.mesh, "DG", 0)
        self.DG1 = df.FunctionSpace(self.mesh, "DG", 1)
        self.V0 = df.FunctionSpace(self.mesh, "P", 1)
        self.V1 = df.VectorFunctionSpace(self.mesh, "P", 1)
        self.V2 = df.TensorFunctionSpace(self.mesh, "P", 1)
        self.ansatz_functions = df.Function(self.function_space)
        self.test_functions = df.TestFunction(self.function_space)

    def set_bio_chem_models(self, input):
        self.bm_model_angiogenesis  = input.bmm.bm_model_angiogenesis
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

    def solve(self):

        def output(time):
            #P_ = df.project(P, self.V2, solver_type="cg")
            #P_vM_ = df.project(calcStress_vonMises(P_), self.V0, solver_type="cg")
            #write_field2output(output_file, df.project(u, self.V1, solver_type="cg"), "u", time)
            #write_field2output(output_file, df.project(p, self.V0, solver_type="cg"), "p", time)
            #write_field2output(output_file, df.project(nS, self.V0, solver_type="cg"), "nS", time)  # , self.eval_points, self.mesh)
            #write_field2output(output_file, df.project(nF, self.V0, solver_type="cg"), "nF", time)
            #write_field2output(output_file, df.project(nSh, self.V0, solver_type="cg"), "nSh", time)  # , self.eval_points, self.mesh)
            #write_field2output(output_file, df.project(nSt, self.V0, solver_type="cg"), "nSt", time)  # , self.eval_points, self.mesh)
            #write_field2output(output_file, df.project(nSn, self.V0, solver_type="cg"), "nSn", time)  # , self.eval_points, self.mesh)
            write_field2output(output_file, df.project(cFn, self.V0, solver_type="cg"), "cFn", time)
            #write_field2output(output_file, df.project(cFt, self.V0, solver_type="cg"), "cFt", time)
            #write_field2output(output_file, df.project(cFv, self.V0, solver_type="cg"), "cFv", time)
            #write_field2output(output_file, df.project(cFa, self.V0, solver_type="cg"), "cFa", time)
            #write_field2output(output_file, df.project(lambdaS, self.V0), "lambdaS", time)
            #write_field2output(output_file, df.project(hatrhoSt, self.V0), "hatrhoSt", time)
            #write_field2output(output_file, df.project(hatrhoFt, self.V0), "hatrhoFt", time)
            #write_field2output(output_file, P_, "stress", time)
            #write_field2output(output_file, P_vM_, "vonMises", time)
            #write_field2output(output_file, df.project(rhoS / rhoFR, self.V0), "VBo3", time)

        prm = df.parameters["form_compiler"]
        prm["quadrature_degree"] = 2

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
        DFn = df.Constant(self.DFn)
        DFt = df.Constant(self.DFt)
        DFv = df.Constant(self.DFv)
        DFa = df.Constant(self.DFa)
        lambdaSh = df.Constant(self.lambdaSh)
        lambdaSt = df.Constant(self.lambdaSt)
        lambdaSn = df.Constant(self.lambdaSn)
        muSh = df.Constant(self.muSh)
        muSt = df.Constant(self.muSt)
        muSn = df.Constant(self.muSn)
        dt = df.Constant(self.dt)

        # general
        output_file = self.output_file

        # Get Ansatz and test functions
        w_n = df.Function(self.function_space)  # old primaries
        u, p, nSh, nSt, nSn, cFn, cFt, cFv, cFa = df.split(self.ansatz_functions)
        _u, _p, _nSh, _nSt, _nSn, _cFn, _cFt, _cFv, _cFa = df.split(self.test_functions)
        u_n, p_n, nSh_n, nSt_n, nSn_n, cFn_n, cFt_n, cFv_n, cFa_n = df.split(w_n)

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
        # Angiogenesis
        hatrhoFv = df.Constant(0.0)
        if self.flag_angiogenesis:
            hatrhoFv = self.bm_model_angiogenesis

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
        hatrhoFa = hatrhoFa_apop

        #######################################
        # express growth via different quantities
        hatrhoS = hatrhoSt + hatrhoSn + hatrhoSh
        hatnS = hatrhoS / rhoS
        hatnSh = hatrhoSh / rhoStR
        hatnSt = hatrhoSt / rhoStR
        hatnSn = hatrhoSn / rhoStR
        hatrhoF = - hatrhoS
        hatnF = hatrhoF / rhoFR
        ##############################################################################

        ##############################################################################
        # Time-dependent fields
        #######################################
        # Calculate velocity
        v = (u - u_n) / dt
        div_v = ufl.inner(D_S, I)
        # Calculate seepage-velocity (wtFS)
        kappaF = kF / gammaFR
        #nFw_F = -kF * ufl.grad(p)
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
        res_VBm21 = ufl.dot(ufl.grad(p), ufl.inv(C_S)) + hatnF / nF * rhoFR * ufl.dot(v, ufl.inv(F_S.T))
        res_VBm2 = J_S * kappaF * ufl.inner(res_VBm21, ufl.grad(_p)) * dx
        res_VBm3 = - J_S * hatnS * (1.0 - rhoS / rhoFR) * _p * dx #ultra empfindlich
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
        nFcFnw_Fn2 = - cFn * kappaF * ufl.dot(ufl.grad(p), ufl.inv(C_S))
        nFcFnw_Fn3 = - cFn * kappaF * hatnF / nF * rhoFR * ufl.dot(v, ufl.inv(F_S.T))
        nFcFnw_Fn = nFcFnw_Fn1 + nFcFnw_Fn2 + nFcFnw_Fn3
        res_CBn1 = J_S * (nF * dcFndt - hatrhoFn / molFn) * _cFn * dx
        res_CBn2 = J_S * cFn * (div_v - hatrhoS / rhoS) * _cFn * dx 
        res_CBn3 = - J_S * ufl.inner(nFcFnw_Fn, ufl.grad(_cFn)) * dx 
        res_CBn = res_CBn1 + res_CBn2 + res_CBn3
        #######################################

        #######################################
        # Concentration balance of solved cancer cells
        nFcFtw_Ft1 = - DFt * ufl.dot(ufl.grad(cFt), ufl.inv(C_S))
        nFcFtw_Ft2 = - cFt * kappaF * ufl.dot(ufl.grad(p), ufl.inv(C_S))
        nFcFtw_Ft3 = - cFt * kappaF * hatnF / nF * rhoFR * ufl.dot(v, ufl.inv(F_S.T))
        nFcFtw_Ft = nFcFtw_Ft1 + nFcFtw_Ft2 + nFcFtw_Ft3
        res_CBt1 = J_S * (nF * dcFtdt - hatrhoFt / molFt) * _cFt * dx
        res_CBt2 = J_S * cFt * (div_v - hatrhoS / rhoS) * _cFt * dx
        res_CBt3 = - J_S * ufl.inner(nFcFtw_Ft, ufl.grad(_cFt)) * dx
        res_CBt = res_CBt1 + res_CBt2 + res_CBt3
        #######################################

        #######################################
        # Concentration balance of solved VEGF
        nFcFvw_Fv1 = - DFv * ufl.dot(ufl.grad(cFv), ufl.inv(C_S))
        nFcFvw_Fv2 = - cFv * kappaF * ufl.dot(ufl.grad(p), ufl.inv(C_S))
        nFcFvw_Fv3 = - cFv * kappaF * hatnF / nF * rhoFR * ufl.dot(v, ufl.inv(F_S.T))
        nFcFvw_Fv = nFcFvw_Fv1 + nFcFvw_Fv2 + nFcFvw_Fv3
        res_CBv1 = J_S * (nF * dcFvdt - hatrhoFv / molFv) * _cFv * dx
        res_CBv2 = J_S * cFv * (div_v - hatrhoS / rhoS) * _cFv * dx
        res_CBv3 = - J_S * ufl.inner(nFcFvw_Fv, ufl.grad(_cFv)) * dx
        res_CBv = res_CBv1 + res_CBv2 + res_CBv3
        #######################################

        #######################################
        # Concentration balance of solved VEGF
        nFcFaw_Fa1 = - DFa * ufl.dot(ufl.grad(cFa), ufl.inv(C_S))
        nFcFaw_Fa2 = - cFa * kappaF * ufl.dot(ufl.grad(p), ufl.inv(C_S))
        nFcFaw_Fa3 = - cFa * kappaF * hatnF / nF * rhoFR * ufl.dot(v, ufl.inv(F_S.T))
        nFcFaw_Fa = nFcFaw_Fa1 + nFcFaw_Fa2 + nFcFaw_Fa3
        res_CBa1 = J_S * (nF * dcFadt - hatrhoFa / molFa) * _cFa * dx
        res_CBa2 = J_S * cFa * (div_v - hatrhoS / rhoS) * _cFa * dx
        res_CBa3 = - J_S * ufl.inner(nFcFaw_Fa, ufl.grad(_cFa)) * dx
        res_CBa = res_CBa1 + res_CBa2 + res_CBa3
        #######################################
        # sum up to total residual
        res_tot = res_LMo + res_VBm + res_VBh + res_VBt + res_VBn + res_CBn + res_CBt + res_CBv + res_CBa
        if not self.n_bound is None:
            res_tot += self.n_bound
        ##############################################################################

        # Define problem solution
        w = self.ansatz_functions
        solver = solv.nonlinvarsolver(res_tot, w, self.d_bound, self.solver_param)

        # Set initial conditions
        w_n.interpolate(self.initial_condition)
        w.interpolate(self.initial_condition)

        # Initialize  and time loop
        t = 0
        output(t)
        print("Initial step is written")
        while t < self.T_end:
            # Increment solution time
            t = t + self.dt

            # Calculate current solution
            n_iter, converged = solver.solve()
            print("Time: {}".format(t), "  ", "Converged in steps: {}".format(n_iter))

            # Output solution
            output(t)

            # Update history fields
            w_n.assign(w)
