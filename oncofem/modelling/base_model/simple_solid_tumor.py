#############################################################
# Growth Paper                                              #
#                                                           #
# Stress-dependent finite growth in soft elastic tissues    #
#                                                           #  
# Author: Marlon Suditsch                                   #
#                                                           #
#############################################################

from oncofem.helper import auxillaries as aux
from oncofem.helper.io import write_field2xdmf
from oncofem.modelling.base_model.solver import nonlinvarsolver as solv

import dolfin
import ufl


class InitialCondition(dolfin.UserExpression):  # UserExpression instead of Expression
    def __init__(self, subdomains, nS_0S, **kwargs):
        super().__init__(**kwargs)
        self.subdomains = subdomains
        self.nS_0S = nS_0S

    def eval_cell(self, values, x, cell):
        values[0] = 0.0  # u_x
        values[1] = 0.0  # u_y
        values[2] = 0.0  # u_z
        values[3] = 0.0  # p
        values[4] = self.nS_0S  # nS

    def value_shape(self):
        return (5,)


class InitialConditionInternals(dolfin.UserExpression):  # UserExpression instead of Expression
    def __init__(self, subdomains, growthArea, hatrhoS, **kwargs):
        super().__init__(**kwargs)
        self.subdomains = subdomains
        self.growthArea = growthArea
        self.hatrhoS = hatrhoS

    def eval_cell(self, values, x, cell):
        values[0] = 0.0  # hatrhoS
        for area in self.growthArea:
            if self.subdomains[cell.index] == area:
                values[0] = self.hatrhoS


class BaseSimpleSolidTumor:
    """
    t.b.d.
    """

    def __init__(self):
        self.flag_actConf = True
        self.flag_defSplit = False
        self.output_file = None

        self.finite_element = None
        self.function_space = None
        self.ansatz_function = None
        self.test_function = None
        self.internal_function_spaces = None
        self.V0 = None
        self.V1 = None
        self.V2 = None

        self.type_u = None
        self.type_p = None
        self.type_nS = None
        self.order_u = None
        self.order_p = None
        self.order_nS = None
        self.mesh = None
        self.domain = None
        self.growthArea = None

        self.dx = None
        self.n_bound = None
        self.d_bound = None
        self.initial_condition = None
        self.internal_condition = None
        self.hatrhoS = None

        self.rhoSR = None
        self.rhoFR = None
        self.lambdaS = None
        self.muS = None
        self.gammaFR = None
        self.nS_0S = None
        self.kF_0S = None

        self.solver_param = None

        self.dt = None
        self.T_end = None

    def set_initial_condition(self):
        self.initial_condition = InitialCondition(self.domain, self.nS_0S)
        self.internal_condition = InitialConditionInternals(self.domain, self.growthArea, self.hatrhoS)

    def set_boundaries(self, d_bound, n_bound):
        self.d_bound = d_bound
        self.n_bound = n_bound

    def set_param(self, input):
        """
        sets parameter needed for model class
        """
        self.eval_points = input.param.gen.eval_points
        self.output_file = input.param.gen.output_file
        self.flag_defSplit = input.param.gen.flag_defSplit
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
        self.hatrhoS = dolfin.Constant(input.param.mat.hatrhoS)
        self.rhoSR = dolfin.Constant(input.param.mat.rhoSR)
        self.rhoFR = dolfin.Constant(input.param.mat.rhoFR)
        self.lambdaS = dolfin.Constant(input.param.mat.lambdaS)
        self.muS = dolfin.Constant(input.param.mat.muS)
        self.gammaFR = dolfin.Constant(input.param.mat.gammaFR)
        self.nS_0S = dolfin.Constant(input.param.mat.nS_0S)
        self.kF_0S = dolfin.Constant(input.param.mat.kF_0S)
        self.solver_param = input.param.fem.solver_param
        self.dt = input.param.time.dt
        self.T_end = input.param.time.T_end

    def set_function_spaces(self):
        """
        sets function space for primary variables u, p, nS and for internal variables
        """
        element_u = dolfin.VectorElement(self.type_u, self.mesh.ufl_cell(), self.order_u)  # defines vector approximation for displacement
        element_p = dolfin.FiniteElement(self.type_p, self.mesh.ufl_cell(), self.order_p)  # defines scalar approximation for pressure
        element_nS = dolfin.FiniteElement(self.type_nS, self.mesh.ufl_cell(), self.order_nS)  # defines scalar approximation for pressure
        self.finite_element = dolfin.MixedElement(element_u, element_p, element_nS)  # assemble element
        self.function_space = dolfin.FunctionSpace(self.mesh, self.finite_element)
        self.V0 = dolfin.FunctionSpace(self.mesh, "P", 1)
        self.V1 = dolfin.VectorFunctionSpace(self.mesh, "P", 1)
        self.V2 = dolfin.TensorFunctionSpace(self.mesh, "P", 1)
        self.ansatz_function = dolfin.Function(self.function_space)
        self.test_function = dolfin.TestFunction(self.function_space)

    def solve(self):

        def output(w, time):
            u_, p_, nS_ = w.split()
            write_field2xdmf(self.output_file, u_, "u", time)  #, self.eval_points, self.mesh)
            write_field2xdmf(self.output_file, dolfin.project(aux.norm(u_), self.V0), "u_av", time)  #, self.eval_points, self.mesh)
            write_field2xdmf(self.output_file, p_, "p", time)  #, self.eval_points, self.mesh)
            write_field2xdmf(self.output_file, nS_, "nS", time)  #, self.eval_points, self.mesh)
            write_field2xdmf(self.output_file, dolfin.project(nF, self.V0), "nF", time)  # , self.eval_points, self.mesh)
            write_field2xdmf(self.output_file, dolfin.project(hatrhoS, self.V0), "hatrhoS", time)

        prm = dolfin.parameters["form_compiler"]
        prm["quadrature_degree"] = 2

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
        cond = ufl.sqrt(hatnF*hatnF)
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
            #print("Time: {}".format(t), "  ", "Converged in steps: {}".format(n_iter))

            # Output solution
            output(w, t)

            w_n.assign(w)
