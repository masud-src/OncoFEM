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

import oncofem.helper.auxillaries as aux
import oncofem.modelling.base_model.solver as solv
import oncofem.modelling.base_model.continuum_mechanics.constitutives as const
import oncofem.modelling.base_model.continuum_mechanics.kinematics as kin
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

        if True:
            values[0] = 0.0  # u_x
            values[1] = 0.0  # u_y
            values[2] = 0.0  # u_z
            values[3] = 0.0  # p
            values[4] = 0.5  # nSh
            values[5] = 0.0  # nSt
            values[6] = 0.0  # nSn
            values[7] = 1.0e-2  # cFn
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
            values[7] = 1.0e-2  # cFn
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


def verhulst_growth(field, kappa, max_value):
    return field * kappa * (1 - field / max_value)

#############################################################
#                                                           #
#  Helper functions                                         #
#                                                           #
#############################################################
class Glioblastoma:

    def __init__(self):
        # General infos
        self.output_file = None
        self.flag_VEGF = True
        self.flag_proliferation = True
        self.flag_metabolism = True
        self.flag_apoptose = True
        self.flag_necrosis = True
        self.flag_defSplit = True

        self.finite_element = None
        self.function_space = None
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
        self.flag_VEGF = input.param.gen.flag_VEGF
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
        element_u = df.VectorElement(self.type_u, self.mesh.ufl_cell(), self.order_u)
        element_p = df.FiniteElement(self.type_p, self.mesh.ufl_cell(), self.order_p)
        element_nSh = df.FiniteElement(self.type_nSh, self.mesh.ufl_cell(), self.order_nSh)
        element_nSt = df.FiniteElement(self.type_nSt, self.mesh.ufl_cell(), self.order_nSt)
        element_nSn = df.FiniteElement(self.type_nSn, self.mesh.ufl_cell(), self.order_nSn)
        element_cFn = df.FiniteElement(self.type_cFn, self.mesh.ufl_cell(), self.order_cFn)
        element_cFt = df.FiniteElement(self.type_cFt, self.mesh.ufl_cell(), self.order_cFt)
        element_cFv = df.FiniteElement(self.type_cFv, self.mesh.ufl_cell(), self.order_cFv)
        element_cFa = df.FiniteElement(self.type_cFa, self.mesh.ufl_cell(), self.order_cFa)
        self.finite_element = df.MixedElement(element_u, element_p, element_nSh, element_nSt, 
                                              element_nSn, element_cFn, element_cFt, element_cFv, element_cFa)
        self.function_space = df.FunctionSpace(self.mesh, self.finite_element)
        self.V0 = df.FunctionSpace(self.mesh, "P", 1)
        self.V1 = df.VectorFunctionSpace(self.mesh, "P", 1)
        self.V2 = df.TensorFunctionSpace(self.mesh, "P", 1)

    def solve(self):

        def output(time):
            T_ = df.project(T, self.V2)
            T_vM_ = df.project(const.calcStress_vonMises(T_), self.V0)
            write_field2output(output_file, df.project(u, self.V1), "u", time)
            write_field2output(output_file, df.project(p, self.V0), "p", time)
            write_field2output(output_file, df.project(nS, self.V0), "nS", time)  # , self.eval_points, self.mesh)
            write_field2output(output_file, df.project(nF, self.V0), "nF", time)
            write_field2output(output_file, df.project(nSh, self.V0), "nSh", time)  # , self.eval_points, self.mesh)
            write_field2output(output_file, df.project(nSt, self.V0), "nSt", time)  # , self.eval_points, self.mesh)
            write_field2output(output_file, df.project(nSn, self.V0), "nSn", time)  # , self.eval_points, self.mesh)
            write_field2output(output_file, df.project(cFn, self.V0), "cFn", time)
            write_field2output(output_file, df.project(cFt, self.V0), "cFt", time)
            write_field2output(output_file, df.project(cFv, self.V0), "cFv", time)
            write_field2output(output_file, df.project(cFa, self.V0), "cFa", time)
            write_field2output(output_file, df.project(lambdaS, self.V0), "lambdaS", time)
            write_field2output(output_file, df.project(hatrhoSt, self.V0), "hatrhoSt", time)
            write_field2output(output_file, df.project(hatrhoFt, self.V0), "hatrhoFt", time)
            write_field2output(output_file, T_, "stress", time)
            write_field2output(output_file, T_vM_, "vonMises", time)
            write_field2output(output_file, df.project(rhoS / rhoFR, self.V0), "VBo3", time)

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

        # Time-dependent Parameters
        T_end = df.Constant(self.T_end)
        dt = df.Constant(self.dt)

        # general
        output_file = self.output_file


        # Store intern variables
        w_n = df.Function(self.function_space)  # old primaries 
        hatrhoSh = df.Function(self.V0)
        hatrhoSt = df.Function(self.V0)
        hatrhoSn = df.Function(self.V0)
        hatrhoFn = df.Function(self.V0)
        hatrhoFv = df.Function(self.V0)
        hatrhoFa = df.Function(self.V0)
        nF = df.Function(self.V0)

        # Get Ansatz and test functions
        w = df.Function(self.function_space)
        _w = df.TestFunction(self.function_space)
        u, p, nSh, nSt, nSn, cFn, cFt, cFv, cFa = df.split(w)
        u_n, p_n, nSh_n, nSt_n, nSn_n, cFn_n, cFt_n, cFv_n, cFa_n = df.split(w_n)
        _u, _p, _nSh, _nSt, _nSn, _cFn, _cFt, _cFv, _cFa = df.split(_w)

        dx = self.dx

        # Kinematics
        I = ufl.Identity(len(u))
        F_S = I + ufl.grad(u)
        F_Sn = I + ufl.grad(u_n)
        J_S = ufl.det(F_S)
        C_S = F_S.T * F_S
        B_S = F_S * F_S.T
        E = kin.calcStrain_GreenLagrange(u)
        E_n = kin.calcStrain_GreenLagrange(u_n)
        dF_Sdt = (F_S - F_Sn) / dt
        L_S = dF_Sdt * ufl.inv(F_S)
        D_S = (L_S + L_S.T) / 2.0

        # Calculate volume fractions
        nS = nSh + nSt + nSn
        nS_n = nSh_n + nSt_n + nSn_n
        rhoS = (nSh * rhoStR + nSt * rhoStR + nSn * rhoStR) / nS
        nF = 1.0 - nS

        ##############################################################################
        # Calculate growth terms
        hatrhoFt = ufl.conditional(ufl.le(cFt, 0.008), verhulst_growth(cFt, 1e-2, 0.008), 0) 
        hatrhoFn = - 0.5e-2 * cFt - 0.5e-3 * nSt
        #hatrhoFv.assign(df.project(0, self.V0))
        #hatrhoFa.assign(df.project(0, self.V0))
        hatrhoSt = ufl.conditional(ufl.ge(cFt, 0.0055), 2.0, 0.0)

        ##############################################################################

        hatrhoS = hatrhoSt + hatrhoSn + hatrhoSh
        hatnS = hatrhoS / rhoS
        hatnSh = hatrhoSh / rhoStR
        hatnSt = hatrhoSt / rhoStR
        hatnSn = hatrhoSn / rhoStR
        hatrhoF = - hatrhoS
        hatnF = hatrhoF / rhoFR

        # Calculate storage terms
        dnShdt = (nSh - nSh_n) / dt
        dnStdt = (nSt - nSt_n) / dt
        dnSndt = (nSn - nSn_n) / dt
        dnSdt = (nS - nS_n) / dt
        dnFdt = - dnSdt
        dcFndt = (cFn - cFn_n) / dt
        dcFtdt = (cFt - cFt_n) / dt
        dcFvdt = (cFv - cFv_n) / dt
        dcFadt = (cFa - cFa_n) / dt

        # Calculate velocity
        v = (u - u_n) / dt
        div_v = ufl.inner(D_S, I)

        # Calculate seepage-velocity (wtFS)
        kappaF = ufl.conditional(ufl.ge(nF, 0.0), (kF * nF * nF) / (gammaFR * nF * nF + kF * hatnF * rhoFR), kF)
        nFw_F = -kF * ufl.grad(p)

        # Calculate Diffusion
        nFcFnw_Fn = -DFn * ufl.grad(cFn) + cFn * nFw_F
        nFcFtw_Ft = -DFt * ufl.grad(cFt) + cFt * nFw_F
        nFcFvw_Fv = -DFv * ufl.grad(cFv) + cFv * nFw_F
        nFcFaw_Fa = -DFa * ufl.grad(cFa) + cFa * nFw_F

        # Calculate Stress
        lambdaS = (lambdaSh * nSh + lambdaSt * nSt + lambdaSn * nSn) / (nSh + nSt + nSn)
        muS = (muSh * nSh + muSt * nSt + muSn * nSn) / (nSh + nSt + nSn)
        TS_E = (muS * (B_S - I) + lambdaS * ufl.ln(J_S) * I) / J_S
        T = TS_E - p * I
        P = J_S * T * ufl.inv(F_S.T)

        ##############################################################################
        # Define weak forms
        #######################################
        # Momentum balance of overall aggregate
        res_LMo1 = ufl.inner(P, ufl.grad(_u)) * dx
        res_LMo2 = - J_S * ((hatrhoS + hatrhoF) + (kappaF * hatrhoF * hatrhoF) / (nF * nF)) * ufl.inner(v, _u) * dx
        res_LMo3 = - J_S * hatrhoF * kappaF / nF * ufl.inner(ufl.dot(ufl.grad(p), ufl.inv(F_S)), _u) * dx 
        res_LMo = res_LMo1 + res_LMo2 + res_LMo3
        #######################################

        #res_VBm = dnFdt * _p * dx - ufl.inner(nFw_F, ufl.grad(_p)) * dx + nF * div_v * _p * dx - hatrhoF / rhoFR * _p * dx
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
        #res_VBh = dnShdt * _nSh * dx - nSh * div_v * _nSh * dx - hatnSh * _nSh * dx
        #######################################

        #######################################
        # Volume balance of tumor cells
        res_VBt1 = J_S * (dnStdt - hatnSt) * _nSt * dx
        res_VBt2 = J_S * nSt * div_v * _nSt * dx
        res_VBt = res_VBt1 + res_VBt2
        #res_VBt = dnStdt * _nSt * dx - nSt * div_v * _nSt * dx - hatnSt * _nSt * dx
        #######################################

        #######################################
        # Volume balance of necrotic cells
        res_VBn1 = J_S * (dnSndt - hatnSn) * _nSt * dx
        res_VBn2 = J_S * nSt * div_v * _nSt * dx
        res_VBn = res_VBn1 + res_VBn2
        res_VBn = dnSndt * _nSn * dx - nSn * div_v * _nSn * dx - hatnSn * _nSn * dx
        #######################################

        #######################################
        # Concentration balance of solved cancer cells
        res_CBt11 = DFt * ufl.dot(ufl.grad(cFt), ufl.inv(C_S))
        res_CBt12 = - cFt * kappaF * ufl.dot(ufl.grad(p), ufl.inv(C_S))
        res_CBt13 = - cFt * kappaF * hatnF / nF * rhoFR * ufl.dot(v, ufl.inv(F_S.T))
        res_CBt1 = - J_S * ufl.inner(res_CBt11, ufl.grad(_cFt)) * dx
        res_CBt2 = J_S * (nF * dcFtdt - cFt * hatnS / rhoS - hatrhoFt / molFt) * _cFt * dx
        res_CBt3 = J_S * cFt * div_v * _cFt * dx
        res_CBt = res_CBt1 + res_CBt2 + res_CBt3
        #######################################
        res_CBt = nF * dcFtdt * _cFt * dx - ufl.inner(nFcFtw_Ft, ufl.grad(_cFt)) * dx + cFt * (div_v - hatrhoS / rhoS) * _cFt * dx - hatrhoFt / molFt * _cFt * dx

        res_CBn = nF * dcFndt * _cFn * dx - ufl.inner(nFcFnw_Fn, ufl.grad(_cFn)) * dx + cFn * (div_v - hatrhoS / rhoS) * _cFn * dx - hatrhoFn / molFn * _cFn * dx
        res_CBv = nF * dcFvdt * _cFv * dx - ufl.inner(nFcFvw_Fv, ufl.grad(_cFv)) * dx + cFv * (div_v - hatrhoS / rhoS) * _cFv * dx - hatrhoFv / molFv * _cFv * dx
        res_CBa = nF * dcFadt * _cFa * dx - ufl.inner(nFcFaw_Fa, ufl.grad(_cFa)) * dx + cFa * (div_v - hatrhoS / rhoS) * _cFa * dx - hatrhoFa / molFa * _cFa * dx

        res_tot = res_LMo + res_VBm + res_VBh + res_VBt + res_VBn + res_CBn + res_CBt + res_CBv + res_CBa# - ip.geom.n_bound

        # Define problem solution
        solver = solv.nonlinvarsolver(res_tot, w, self.d_bound, self.solver_param)

        # Set initial conditions
        w_n.interpolate(self.initial_condition)
        w.interpolate(self.initial_condition)

        # Initialize solution time
        t = 0

        # Initial step output
        output(t)

        # Time loop
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
