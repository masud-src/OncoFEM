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
            values[7] = 0.0  # cFt
        if self.subdomains[cell.index] == 5:
            values[0] = 0.0  # u_x
            values[1] = 0.0  # u_y
            values[2] = 0.0  # u_z
            values[3] = 0.0  # p
            values[4] = 0.5  # nSh
            values[5] = 0.0  # nSt
            values[6] = 0.0  # nSn
            values[7] = 6.0e-3  # cFt

    def value_shape(self):
        return (8,)


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
        self.type_cFt = input.param.fem.type_cFt
        self.order_u = input.param.fem.order_u
        self.order_p = input.param.fem.order_p
        self.order_nSh = input.param.fem.order_nSh
        self.order_nSt = input.param.fem.order_nSt
        self.order_nSn = input.param.fem.order_nSn
        self.order_cFt = input.param.fem.order_cFt
        self.mesh = input.geom.mesh
        self.domain = input.geom.domain
        self.dx = input.geom.dx
        self.n_bound = input.geom.n_bound
        self.d_bound = input.geom.d_bound
        self.rhoShR = df.Constant(input.param.mat.rhoShR)
        self.rhoStR = df.Constant(input.param.mat.rhoStR)
        self.rhoSnR = df.Constant(input.param.mat.rhoSnR)
        self.rhoFR = df.Constant(input.param.mat.rhoFR)
        self.molFt = df.Constant(input.param.mat.molFt)
        self.kF = df.Constant(input.param.mat.kF)
        self.DFt = df.Constant(input.param.mat.DFt)
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
        element_cFt = df.FiniteElement(self.type_cFt, self.mesh.ufl_cell(), self.order_cFt)
        self.finite_element = df.MixedElement(element_u, element_p, element_nSh, element_nSt, element_nSn, element_cFt)
        self.function_space = df.FunctionSpace(self.mesh, self.finite_element)
        self.V0 = df.FunctionSpace(self.mesh, "P", 1)
        self.V1 = df.VectorFunctionSpace(self.mesh, "P", 1)
        self.V2 = df.TensorFunctionSpace(self.mesh, "P", 1)

    def solve(self):

        def output(time):
            nF_ = df.project(nF, self.V0)
            hatrhoSt_ = df.project(hatrhoSt, self.V0)
            hatrhoFt_ = df.project(hatrhoFt, self.V0)
            T_ = df.project(T, self.V2)
            T_vM_ = df.project(const.calcStress_vonMises(T_), self.V0)
            write_field2output(output_file, dolfin.project(u, self.V1), "u", time)
            write_field2output(output_file, dolfin.project(p, self.V0), "p", time)
            write_field2output(output_file, dolfin.project(cFt, self.V0), "cFt", time)
            write_field2output(output_file, df.project(nS, self.V0), "nS", time)  # , self.eval_points, self.mesh)
            write_field2output(output_file, df.project(nSh, self.V0), "nSh", time)  # , self.eval_points, self.mesh)
            write_field2output(output_file, df.project(nSt, self.V0), "nSt", time)  # , self.eval_points, self.mesh)
            write_field2output(output_file, df.project(nSn, self.V0), "nSn", time)  # , self.eval_points, self.mesh)
            write_field2output(output_file, nF_, "nF", time)
            write_field2output(output_file, hatrhoSt_, "hatrhoSt", time)
            write_field2output(output_file, hatrhoFt_, "hatrhoFt", time)
            write_field2output(output_file, T_, "stress", time)
            write_field2output(output_file, T_vM_, "vonMises", time)

        prm = df.parameters["form_compiler"]
        prm["quadrature_degree"] = 2

        # Material Parameters and Parameters that go into weak form
        rhoShR = df.Constant(self.rhoShR)
        rhoStR = df.Constant(self.rhoStR)
        rhoSnR = df.Constant(self.rhoSnR)
        rhoFR = df.Constant(self.rhoFR)
        molFt = df.Constant(self.molFt)
        kF = df.Constant(self.kF)
        DFt = df.Constant(self.DFt)
        lambdaSh = df.Constant(self.lambdaSh)
        lambdaSt = df.Constant(self.lambdaSt)
        lambdaSn = df.Constant(self.lambdaSn)
        muSh = df.Constant(self.muSh)
        muSt = df.Constant(self.muSt)
        muSn = df.Constant(self.muSn)

        # Time-dependent Parameters
        T_end = self.T_end
        dt = self.dt

        # general
        output_file = self.output_file

        # **********************************************************************
        # --- Calculate solution -----------------------------------------------
        # **********************************************************************
        # Store intern variables
        w_n = df.Function(self.function_space)  # old primaries 
        hatrhoSh = df.Function(self.V0)
        hatrhoSt = df.Function(self.V0)
        hatrhoSn = df.Function(self.V0)
        hatrhoFt = df.Function(self.V0)
        nF = df.Function(self.V0)

        u_n, p_n, nSh_n, nSt_n, nSn_n, cFt_n = df.split(w_n)

        # Get Ansatz and test functions
        w = df.Function(self.function_space)
        _w = df.TestFunction(self.function_space)
        u, p, nSh, nSt, nSn, cFt = df.split(w)
        _u, _p, _nSh, _nSt, _nSn, _cFt = df.split(_w)

        dx = self.dx

        # Kinematics
        F = kin.calc_defGrad(u)
        J = kin.calc_detDefGrad(u)
        E = kin.calcStrain_GreenLagrange(u)
        E_n = kin.calcStrain_GreenLagrange(u_n)

        # Calculate volume fractions
        nS = nSh + nSt + nSn
        nS_n = nSh_n + nSt_n + nSn_n
        rhoS = nSh * rhoStR + nSt * rhoStR + nSn * rhoStR
        nF = 1.0 - nS
        hatrhoS = hatrhoSt
        hatnS = hatrhoS / rhoS
        hatnSh = hatrhoSh / rhoStR
        hatnSt = hatrhoSt / rhoStR
        hatnSn = hatrhoSn / rhoStR
        hatrhoF = - hatrhoS
        hatnF = hatrhoF / rhoFR

        # Calculate storage terms
        dcFtdt = (cFt - cFt_n) / dt
        dnShdt = (nSh - nSh_n) / dt
        dnStdt = (nSt - nSt_n) / dt
        dnSndt = (nSn - nSn_n) / dt
        dnSdt = (nS - nS_n) / dt
        dnFdt = - dnSdt

        # Calculate velocity
        div_v = (1 / dt) * ufl.tr(E - E_n)

        # Calculate seepage-velocity (wtFS)
        nFw_F = -kF * ufl.grad(p)

        # Calculate Diffusion
        nFcFtw_Ft = -DFt * ufl.grad(cFt) + cFt * nFw_F

        # Calculate Stress
        T = const.calcStressExtra_NeoHookean_PiolaKirchhoff2_lin(u, p, lambdaSh, muSh)

        # Define weak forms
        res_BLM = ufl.inner(T, ufl.grad(_u)) * dx
        res_BMI = dnFdt * _p * dx - ufl.inner(nFw_F, ufl.grad(_p)) * dx + nF * div_v * _p * dx - hatrhoF / rhoFR * _p * dx
        res_VBh = dnShdt * _nSh * dx - nSh * div_v * _nSh * dx - hatnSh * _nSh * dx
        res_VBt = dnStdt * _nSt * dx - nSt * div_v * _nSt * dx - hatnSt * _nSt * dx
        res_VBn = dnSndt * _nSn * dx - nSn * div_v * _nSn * dx - hatnSn * _nSn * dx
        res_BCT = nF * dcFtdt * _cFt * dx - ufl.inner(nFcFtw_Ft, ufl.grad(_cFt)) * dx + cFt * (div_v - hatrhoS / rhoStR) * _cFt * dx - hatrhoFt / molFt * _cFt * dx

        res_tot = res_BLM + res_BMI + res_VBh + res_VBt + res_VBn + res_BCT# - ip.geom.n_bound

        # Define problem solution
        solver = solv.nonlinvarsolver(res_tot, w, self.d_bound, self.solver_param)

        # Set initial conditions
        w_n.interpolate(self.initial_condition)
        w.interpolate(self.initial_condition)
        hatrhoFt.assign(dolfin.project(cFt, self.V0))
        hatrhoS.assign(dolfin.project(cFt, self.V0))

        # Initialize solution time
        t = 0

        # Initial step output
        output(t)

        # Time loop
        while t < T_end:

            # Increment solution time
            t = t + dt

            # Print current time
            df.info("Time: {}".format(t))

            # Calculate current solution
            solver.solve()

            # Output solution
            output(t)

            # Update history fields
            w_n.assign(w)
