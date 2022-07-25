#############################################################
# Porous Media                                              #
# 2 Phases:                                                 #
#           - Incompressible Solid Phase - phi^S            #
#           - Incompressible Fluid Phase - phi^F            #
#                                                           #
# Weak Forms:                                               #
#           - Momentum Balance of Overall Aggregate         #
#                                                           #
#                                                           #
#           - Volume Balance of Overall Aggregate           #
#                                                           # 
#                                                           #
#############################################################
import dolfin
import dolfin as df
import ufl

from oncofem.helper.io import write_field2output
from oncofem.modelling.base_model.solver import nonlinvarsolver as nlsolv

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
        if self.subdomains[cell.index] == 5:
            values[0] = 0.0  # u_x
            values[1] = 0.0  # u_y
            values[2] = 0.0  # u_z
            values[3] = 0.5  # nSh
            values[4] = 0.0  # nSt
            values[5] = 0.0  # nSn
            values[6] = 0.0  # p
            values[7] = 0.0  # cFt
        if self.subdomains[cell.index] == 6:
            values[0] = 0.0  # u_x
            values[1] = 0.0  # u_y
            values[2] = 0.0  # u_z
            values[3] = 0.5  # nSh
            values[4] = 0.0  # nSt
            values[5] = 0.0  # nSn
            values[6] = 0.0  # p
            values[7] = 0.1  # cFt

    def value_shape(self):
        return (8,)


class InitialConditionInternals(dolfin.UserExpression):  # UserExpression instead of Expression
    def __init__(self, subdomains, growthArea, **kwargs):
        super().__init__(**kwargs)
        self.subdomains = subdomains
        self.growthArea = growthArea

    def eval_cell(self, values, x, cell):
        values[0] = 0.0  # hatrhoS
        for area in self.growthArea:
            if self.subdomains[cell.index] == area:
                values[0] = 1e-4

class InitialnS(df.UserExpression):  # UserExpression instead of Expression
    def __init__(self, subdomains, nS_0S, **kwargs):
        super().__init__(**kwargs)
        self.subdomains = subdomains
        self.nS_0S = nS_0S

    def eval_cell(self, values, x, cell):
        values[0] = self.nS_0S

#############################################################
#                                                           #
#  Helper functions                                         #
#                                                           #
#############################################################
class GlioblastomaX:
    """
    t.b.d
    """

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
        self.type_cFt = None
        self.order_u = None
        self.order_p = None
        self.order_nSh = None
        self.order_nSt = None
        self.order_nSn = None
        self.order_cFt = None
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
        self.molFt = None
        self.kF = None
        self.DFt = None
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
        self.internal_condition = InitialConditionInternals(self.domain, self.growthArea)
        #self.initial_nS = InitialnS(self.domain, self.nS_0S)

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
        self.type_cFt = input.param.fem.type_cFt
        self.order_u = input.param.fem.order_u
        self.order_p = input.param.fem.order_p
        self.order_nSh = input.param.fem.order_nSh
        self.order_nSt = input.param.fem.order_nSt
        self.order_nSn = input.param.fem.order_nSn
        self.order_cFt = input.param.fem.order_cFt
        self.mesh = input.geom.mesh
        self.growthArea = input.geom.growthArea
        self.domain = input.geom.domain
        self.dx = input.geom.dx
        self.n_bound = input.geom.n_bound
        self.d_bound = input.geom.d_bound
        self.rhoShR = df.Constant(input.param.mat.rhoShR)
        self.rhoStR = df.Constant(input.param.mat.rhoStR)
        self.rhoSnR = df.Constant(input.param.mat.rhoSnR)
        self.rhoFR = df.Constant(input.param.mat.rhoFR)
        self.molFt = df.Constant(input.param.mat.molFt)
        self.gammaFR = df.Constant(input.param.mat.gammaFR)
        self.kF = df.Constant(input.param.mat.kF)
        self.DFt = df.Constant(input.param.mat.DFt)
        self.lambdaSh = df.Constant(input.param.mat.lambdaSh)
        self.lambdaSt = df.Constant(input.param.mat.lambdaSt)
        self.lambdaSn = df.Constant(input.param.mat.lambdaSt)
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
        element_nSh = df.FiniteElement(self.type_nSh, self.mesh.ufl_cell(), self.order_nSh)
        element_nSt = df.FiniteElement(self.type_nSt, self.mesh.ufl_cell(), self.order_nSt)
        element_nSn = df.FiniteElement(self.type_nSn, self.mesh.ufl_cell(), self.order_nSn)
        element_p = df.FiniteElement(self.type_p, self.mesh.ufl_cell(), self.order_p)
        element_cFt = df.FiniteElement(self.type_cFt, self.mesh.ufl_cell(), self.order_cFt)
        self.finite_element = df.MixedElement(element_u, element_nSh, element_nSt, element_nSn, element_p, element_cFt)
        self.function_space = df.FunctionSpace(self.mesh, self.finite_element)
        self.V0 = df.FunctionSpace(self.mesh, "P", 1)
        self.V1 = df.VectorFunctionSpace(self.mesh, "P", 1)
        self.V2 = df.TensorFunctionSpace(self.mesh, "P", 1)

    def solve(self):

        def output(w, time):
            u, nSh, nSt, nSn, p, cFt = w.split()
            write_field2output(output_file, u, "u", time)
            write_field2output(output_file, nSh, "nSh", time)
            write_field2output(output_file, nSt, "nSt", time)
            write_field2output(output_file, nSn, "nSn", time)
            write_field2output(output_file, df.project(nF, self.V0), "nF", time)
            write_field2output(output_file, p, "p", time)
            write_field2output(output_file, cFt, "cFt", time)
            write_field2output(output_file, df.project(hatrhoFt, self.V0), "hatrhoFt", time)
            #write_field2output(output_file, nSt, "nSt", time)  # , self.eval_points, self.mesh)     # write_field2output(output_file, T_vM_, "vonMises", time)

        prm = df.parameters["form_compiler"]
        prm["quadrature_degree"] = 2

        # Material Parameters and Parameters that go into weak form
        rhoShR = df.Constant(self.rhoShR)
        rhoStR = df.Constant(self.rhoStR)
        rhoSnR = df.Constant(self.rhoSnR)
        rhoFR = df.Constant(self.rhoFR)
        molFt = df.Constant(self.molFt)
        kF = df.Constant(self.kF)
        gammaFR = df.Constant(self.gammaFR)
        DFt = df.Constant(self.DFt)
        lambdaSh = df.Constant(self.lambdaSh)
        lambdaSt = df.Constant(self.lambdaSt)
        lambdaSn = df.Constant(self.lambdaSt)
        muSh = df.Constant(self.muSh)
        muSt = df.Constant(self.muSt)
        muSn = df.Constant(self.muSt)

        # Time-dependent Parameters
        T_end = self.T_end
        dt = self.dt

        # general
        output_file = self.output_file

        # Store intern variables
        w_n = df.Function(self.function_space)  # old primaries 
        hatnSh = df.Function(self.V0)
        hatnSt = df.Function(self.V0)
        hatnSn = df.Function(self.V0)
        hatrhoS = df.Function(self.V0)
        hatrhoFt = df.Function(self.V0)
        time = dolfin.Constant(0)

        # Get Ansatz and test functions
        w = df.Function(self.function_space)
        _w = df.TestFunction(self.function_space)
        u, nSh, nSt, nSn, p, cFt = df.split(w_n)
        u_n, nSh_n, nSt_n, nSn_n, p_n, cFt_n = df.split(w_n)
        _u, _nSh, _nSt, _nSn, _p, _cFt = df.split(_w)

        dx = self.dx

        # Calculate volume fractions
        nS = nSh + nSt + nSn
        rhoS = nSh * rhoShR + nSt * rhoStR + nSn * rhoSnR
        nS_n = nSh_n + nSt_n + nSn_n
        nF = 1.0 - nS
        hatnS = hatrhoS / rhoS
        hatrhoF = - hatrhoS
        hatnF = hatrhoF / rhoFR

        ##############################################################################
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
        dF_Sdt = (F_S - F_Sn) / dt
        L_S = dF_Sdt * ufl.inv(F_S)
        D_S = (L_S + L_S.T) / 2.0

        # Calculate velocity
        v = (u - u_n) / dt
        div_v = ufl.inner(D_S, I)
        ##############################################################################

        # Calculate storage terms
        dnShdt = (nSh - nSh_n) / dt
        dnStdt = (nSt - nSt_n) / dt
        dnSndt = (nSn - nSn_n) / dt
        dcFtdt = (cFt - cFt_n) / dt

        ##############################################################################
        #######################################
        # Calculate permeability
        kappaF = ufl.conditional(ufl.ge(hatnF,0.0),(kF * nF * nF)/(gammaFR * nF * nF + kF * hatnF * rhoFR), kF)
        kappaF = kF

        #######################################

        #######################################
        # Calculate Stress
        lambdaS = (lambdaSh * nSh + lambdaSt * nSt + lambdaSn * nSn) / (nSh + nSt + nSn)
        muS = (muSh * nSh + muSt * nSt + muSn * nSn) / (nSh + nSt + nSn)
        TS_E = (muS * (B_SE - I) + lambdaS * ufl.ln(J_SE) * I) / J_SE
        T = TS_E - p * I
        P_S = J_S * T * ufl.inv(F_S.T)
        #######################################
        ##############################################################################

        ##############################################################################
        # Define weak forms
        #######################################
        # Momentum balance of overall aggregate
        res_LMo1 = ufl.inner(P_S, ufl.grad(_u)) * dx
        res_LMo2 = - J_S * ((hatrhoS + hatrhoF) + (kappaF * hatrhoF * hatrhoF) / (nF * nF)) * ufl.inner(v, _u) * dx
        res_LMo3 = - J_S * hatrhoF * kappaF / nF * ufl.inner(ufl.dot(ufl.grad(p), ufl.inv(F_S)), _u) * dx 
        res_LMo = res_LMo1 + res_LMo2 + res_LMo3
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
        res_VBn1 = J_S * (dnSndt - hatnSn) * _nSt * dx
        res_VBn2 = J_S * nSt * div_v * _nSt * dx
        res_VBn = res_VBn1 + res_VBn2
        #######################################

        #######################################
        # Volume balance of the mixture
        res_VBm1 = J_S * div_v * _p * dx
        res_VBm21 = ufl.dot(ufl.grad(p), ufl.inv(C_S)) + hatnF / nF * rhoFR * ufl.dot(v, ufl.inv(F_S.T))
        res_VBm2 = J_S * kappaF * ufl.inner(res_VBm21, ufl.grad(_p)) * dx
        res_VBm3 = - J_S * hatnS * (1.0 - rhoS / rhoFR) * _p * dx
        res_VBm = res_VBm1 + res_VBm2 + res_VBm3
        #######################################

        #######################################
        # Concentration balance of solved components
        res_CBt11 = DFt * ufl.dot(ufl.grad(cFt), ufl.inv(C_S)) 
        res_CBt12 = - cFt * kappaF * ufl.dot(ufl.grad(p), ufl.inv(C_S))
        res_CBt13 = - cFt * kappaF * hatnF / nF * rhoFR * ufl.dot(v, ufl.inv(F_S.T))
        res_CBt1 = - J_S * ufl.inner(res_CBt11, ufl.grad(_cFt)) * dx
        res_CBt2 = J_S * (nF * dcFtdt - cFt * hatnS / rhoS - hatrhoFt / molFt) * _cFt * dx
        res_CBt3 = J_S * cFt * div_v * _cFt * dx
        res_CBt = res_CBt1 + res_CBt2 + res_CBt3
        #######################################

        res_tot = res_LMo + res_VBh + res_VBt + res_VBn + res_VBm + res_CBt  # - ip.geom.n_bound
        ##############################################################################

        # Define problem solution
        solver = nlsolv(res_tot, w, self.d_bound, self.solver_param)

        # Set initial conditions
        w_n.interpolate(self.initial_condition)
        w.interpolate(self.initial_condition)

        # Initialize old step
        t = 0

        output(w, t)

        # Time loop
        while t < T_end:
            # Increment solution time
            t = t + dt

            # Print current time
            df.info("Time: {}".format(t))
            print(t)

            # Calculate current solution
            solver.solve()

            # Output solution
            output(w, t)

            # Update history fields
            w_n.assign(w)
