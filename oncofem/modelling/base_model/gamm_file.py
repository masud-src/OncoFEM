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

import oncofem.helper.auxillaries as aux
import oncofem.modelling.base_model.solver as solv
import oncofem.modelling.base_model.continuum_mechanics.constitutives as const
import oncofem.modelling.base_model.continuum_mechanics.kinematics as kin
from oncofem.helper.io import write_field2xdmf
import dolfin as df
import ufl


#############################################################
#                                                           #
#  Helper functions                                         #
#                                                           #
#############################################################

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
            values[4] = 1.0  # cIn
            values[5] = 0.0  # cIt
            values[6] = 0.0  # cIv
        if self.subdomains[cell.index] > 0:
            values[0] = 0.0  # u_x
            values[1] = 0.0  # u_y
            values[2] = 0.0  # u_z
            values[3] = 0.0  # p
            values[4] = 1.0  # cIn
            values[5] = 1.5e-3  # cIt
            values[6] = 0.0  # cIv
        if self.subdomains[cell.index] == 1:
            values[0] = 0.0  # u_x
            values[1] = 0.0  # u_y
            values[2] = 0.0  # u_z
            values[3] = 0.0  # p
            values[4] = 1.0  # cIn
            values[5] = 4e-3  # 1.15e-13    # cIt
            values[6] = 0.0  # cIv
        if self.subdomains[cell.index] == 2:
            values[0] = 0.0  # u_x
            values[1] = 0.0  # u_y
            values[2] = 0.0  # u_z
            values[3] = 0.0  # p
            values[4] = 1.0  # cIn
            values[5] = 1e-3  # 1.15e-13    # cIt
            values[6] = 0.0  # cIv
        if self.subdomains[cell.index] == 3:
            values[0] = 0.0  # u_x
            values[1] = 0.0  # u_y
            values[2] = 0.0  # u_z
            values[3] = 0.0  # p
            values[4] = 1.0  # cIn
            values[5] = 3e-3  # 1.15e-13    # cIt
            values[6] = 0.0  # cIv
        if self.subdomains[cell.index] == 5:
            values[0] = 0.0  # u_x
            values[1] = 0.0  # u_y
            values[2] = 0.0  # u_z
            values[3] = 0.0  # p
            values[4] = 1.0  # cIn
            values[5] = 3e-3  # 1.15e-13    # cIt
            values[6] = 0.0  # cIv

    def value_shape(self):
        return (7,)


class InitialConditionInternals(df.UserExpression):  # UserExpression instead of Expression
    def __init__(self, subdomains, **kwargs):
        super().__init__(**kwargs)
        self.subdomains = subdomains

    def eval_cell(self, values, x, cell):
        if self.subdomains[cell.index] == 7:
            values[0] = 0.0  # hatnS
        elif self.subdomains[cell.index] == 8:
            values[0] = 1e-2  # hatnS

    # def value_rank(self):
    #    return 1


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
class TPM_2Phase6Component_MAfLMoCOnCOtCOv_BrainTumour:
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
        self.flag_actConf = True

        self.finite_element = None
        self.function_space = None
        self.V0 = None
        self.V1 = None
        self.V2 = None

        self.type_u = None
        self.type_p = None
        self.type_cin = None
        self.type_cit = None
        self.type_civ = None
        self.order_u = None
        self.order_p = None
        self.order_cin = None
        self.order_cit = None
        self.order_civ = None
        self.mesh = None
        self.domain = None
        self.growthArea = None

        self.dx = None
        self.n_bound = None
        self.d_bound = None
        self.initial_condition = None
        self.internal_condition = None

        # Material Parameters
        self.nS_0S = None
        self.nSt_0S = None
        self.cIn_0S = None
        self.cIt_0S = None
        self.rhoShR = None
        self.rhoStR = None
        self.rhoIR = None
        self.molIn = None
        self.molIt = None
        self.molIv = None
        self.kI = None
        self.DIn = None
        self.DIt = None
        self.DIv = None
        self.lambdaSh = None
        self.lambdaSt = None
        self.muSh = None
        self.muSt = None

        # Growth parameters
        self.prolifWarburgFac = None
        self.nutrientCellsMin = None
        self.cIn_Max = None
        self.cIn_min = None
        self.nutrientGrowthFactor = None  # Factor of nutrient consumption
        self.cIn_tresGrowthMin = None  # Survival mode
        self.cIn_tresVEGF = None  # Nutrient min to send out VEGF
        self.nT_tres = None  # Tumour big enough for VEGF
        self.vegf_max = None  # Maximal VEGF concentration
        self.massT_tresMin = None  #
        # _________________ Monod Necrosis __________________________________#
        self.mu_It_necros = None
        self.K_It_necros = None
        self.mu_T_necros = None
        self.K_T_necros = None
        # _________________ Verhulst Prolif and Apoptosis ___________________#
        self.nT_max = None
        self.cIt2nT = None
        self.kappa_It_prolif = None
        self.kappa_It_apop = None
        self.kappa_T_prolif = None
        self.kappa_T_apop = None
        # _________________ Metabolism ______________________________________#
        self.alpha_In_prolif = None
        self.alpha_In_survival = None

        # Time Parameters
        self.time = None
        self.T_end = None
        self.dt = None

        # FEM Paramereters
        self.solver_param = None

    def calc_partial_density(self, vol_frac, effective_density):
        return vol_frac * effective_density

    def calc_mass(self, vol_frac, effective_density, ele_volume):
        return self.calc_partial_density(vol_frac, effective_density) * ele_volume

    def calc_hatrhoIv(self, nSt, nI, rhoIv, cIn):
        """
        Calculates production term of VEGF if carcinoma is already there and if nutrients sinks below a threshold value. 
        Returns density
        """
        return ufl.conditional(ufl.ge(nSt, df.DOLFIN_EPS), ufl.conditional(ufl.le(cIn, self.cIn_tresVEGF), self.vegf_max * nI * self.molIv - rhoIv, 0.0), 0.0)

    def calc_hatrhoIn(self, cIn, massIt, massSt):
        """
            Calculates proliferation term of mobile cancer cells. First checks if enough nutrients there, then only where concenctration is, growth can happen.
        """
        cond1 = ufl.conditional(ufl.gt(massIt, df.DOLFIN_EPS), -self.alpha_In_prolif * massIt, 0.0)
        cond2 = ufl.conditional(ufl.gt(massSt, df.DOLFIN_EPS), -self.alpha_In_prolif * massSt, 0.0)
        cond3 = ufl.conditional(ufl.gt(massIt, df.DOLFIN_EPS), -self.alpha_In_survival * massIt, 0.0)
        cond4 = ufl.conditional(ufl.gt(massSt, df.DOLFIN_EPS), -self.alpha_In_survival * massSt, 0.0)
        cond5 = ufl.conditional(ufl.ge(cIn, self.cIn_tresGrowthMin), cond3 + cond4, 0.0)
        return ufl.conditional(ufl.ge(cIn, self.cIn_tresVEGF), cond1 + cond2, cond5)

    def calc_hatrhoIt_prol(self, cIn, cIt):
        """
        Calculates proliferation term of mobile cancer cells. First checks if enough nutrients there, then only where concenctration is, growth can happen.
        """
        return ufl.conditional(ufl.ge(cIn, self.cIn_tresGrowthMin), ufl.conditional(cIt > 0.0, const.calcGrowth_modVerhulst(cIt, self.molIt, self.kappa_It_prolif, self.cIt2nT, self.dt), 0), 0)

    def calc_hatrhoIt_apop(self, cIt):
        """
            Calculates proliferation term of mobile cancer cells. First checks if enough nutrients there, then only where concenctration is, growth can happen.
        """
        return ufl.conditional(ufl.gt(cIt, 0.0), -kin.calcGrowth_modVerhulst(cIt, self.molIt, self.kappa_It_apop, self.cIt2nT, self.dt), 0)

    def calc_hatrhoIt_nec(self, cIt, cIn, nI):
        """
            Calculates proliferation term of mobile cancer cells. First checks if enough nutrients there, then only where concenctration is, growth can happen.
        """
        return ufl.conditional(ufl.gt(cIt, df.DOLFIN_EPS), -kin.calcGrowth_modMonod(cIt * nI, self.mu_It_necros, self.K_It_necros, cIn, self.cIn_tresGrowthMin), 0.0)

    def calc_hatrhoSt_prolif(self, nSt, cIn):
        """
        Calculates proliferation term of mobile cancer cells. First checks if enough nutrients there, then only where concenctration is, growth can happen.
        """
        return ufl.conditional(ufl.ge(cIn, self.cIn_tresGrowthMin), ufl.conditional(ufl.gt(nSt, 0.0), const.calcGrowth_modVerhulst(nSt, self.rhoStR, self.kappa_T_prolif, self.nT_max, self.dt), 0), 0)

    def calc_hatrhoSt_apop(self, nSt):
        """
        Calculates proliferation term of mobile cancer cells. First checks if enough nutrients there, then only where concenctration is, growth can happen.
        """
        return ufl.conditional(ufl.gt(nSt, 0.0), -const.calcGrowth_modVerhulst(nSt, self.rhoStR, self.kappa_T_apop, self.nT_max, self.dt), 0)

    def calc_hatrhoSt_nec(self, nSt, rhoSt, cIn):
        """
        Calculates proliferation term of mobile cancer cells. First checks if enough nutrients there, then only where concenctration is, growth can happen.
        """
        return ufl.conditional(ufl.gt(nSt, df.DOLFIN_EPS), -const.calcGrowth_modMonod(rhoSt, self.mu_T_necros, self.K_T_necros, cIn, self.cIn_tresGrowthMin), 0.0)

    def set_initial_condition(self):
        self.initial_condition = InitialCondition(self.domain)
        self.internal_condition = InitialConditionInternals(self.domain)
        self.initial_nS = InitialnS(self.domain, self.nS_0S)

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
        self.flag_actConf = input.param.gen.flag_actConf
        self.type_u = input.param.fem.type_u
        self.type_p = input.param.fem.type_p
        self.type_cin = input.param.fem.type_cin
        self.type_cit = input.param.fem.type_cit
        self.type_civ = input.param.fem.type_civ
        self.order_u = input.param.fem.order_u
        self.order_p = input.param.fem.order_p
        self.order_cin = input.param.fem.order_cin
        self.order_cit = input.param.fem.order_cit
        self.order_civ = input.param.fem.order_civ
        self.mesh = input.geom.mesh
        self.domain = input.geom.domain
        self.dx = input.geom.dx
        self.n_bound = input.geom.n_bound
        self.d_bound = input.geom.d_bound
        self.nS_0S = df.Constant(input.param.mat.nS_0S)
        self.nSt_0S = df.Constant(input.param.mat.nSt_0S)
        self.cIn_0S = df.Constant(input.param.mat.cIn_0S)
        self.cIt_0S = df.Constant(input.param.mat.cIt_0S)
        self.rhoShR = df.Constant(input.param.mat.rhoShR)
        self.rhoStR = df.Constant(input.param.mat.rhoStR)
        self.rhoIR = df.Constant(input.param.mat.rhoIR)
        self.molIn = df.Constant(input.param.mat.molIn)
        self.molIt = df.Constant(input.param.mat.molIt)
        self.molIv = df.Constant(input.param.mat.molIv)
        self.kI = df.Constant(input.param.mat.kI)
        self.DIn = df.Constant(input.param.mat.DIn)
        self.DIt = df.Constant(input.param.mat.DIt)
        self.DIv = df.Constant(input.param.mat.DIv)
        self.lambdaSh = df.Constant(input.param.mat.lambdaSh)
        self.lambdaSt = input.param.mat.lambdaSt
        self.muSh = input.param.mat.muSh
        self.muSt = input.param.mat.muSt
        self.prolifWarburgFac = input.param.mat.growth.prolifWarburgFac
        self.nutrientCellsMin = input.param.mat.growth.nutrientCellsMin
        self.cIn_Max = input.param.mat.growth.cIn_Max
        self.cIn_min = input.param.mat.growth.cIn_min
        self.nutrientGrowthFactor = input.param.mat.growth.nutrientGrowthFactor  # Factor of nutrient consumption
        self.cIn_tresGrowthMin = input.param.mat.growth.cIn_tresGrowthMin  # Survival mode
        self.cIn_tresVEGF = input.param.mat.growth.cIn_tresVEGF  # Nutrient min to send out VEGF
        self.nT_tres = input.param.mat.growth.nT_tres  # Tumour big enough for VEGF
        self.vegf_max = input.param.mat.growth.vegf_max  # Maximal VEGF concentration
        self.massT_tresMin = input.param.mat.growth.massT_tresMin  #
        self.mu_It_necros = input.param.mat.growth.mu_It_necros
        self.K_It_necros = input.param.mat.growth.K_It_necros
        self.mu_T_necros = input.param.mat.growth.mu_T_necros
        self.K_T_necros = input.param.mat.growth.K_T_necros
        self.nT_max = input.param.mat.growth.nT_max
        self.cIt2nT = input.param.mat.growth.cIt2nT
        self.kappa_It_prolif = input.param.mat.growth.kappa_It_prolif
        self.kappa_It_apop = input.param.mat.growth.kappa_It_apop
        self.kappa_T_prolif = input.param.mat.growth.kappa_T_prolif
        self.kappa_T_apop = input.param.mat.growth.kappa_T_apop
        self.alpha_In_prolif = input.param.mat.growth.alpha_In_prolif
        self.alpha_In_survival = input.param.mat.growth.alpha_In_survival
        self.T_end = input.param.time.T_end
        self.dt = input.param.time.dt
        self.solver_param = input.param.fem.solver_param

    def set_function_spaces(self):
        """
            sets function space for primary variables u, p, cIn, cIt, cIv and for internal variables
        """
        element_u = df.VectorElement(self.type_u, self.mesh.ufl_cell(), self.order_u)
        element_p = df.FiniteElement(self.type_p, self.mesh.ufl_cell(), self.order_p)
        element_cin = df.FiniteElement(self.type_cin, self.mesh.ufl_cell(), self.order_cin)
        element_cit = df.FiniteElement(self.type_cit, self.mesh.ufl_cell(), self.order_cit)
        element_civ = df.FiniteElement(self.type_civ, self.mesh.ufl_cell(), self.order_civ)
        self.finite_element = df.MixedElement(element_u, element_p, element_cin, element_cit, element_civ)
        self.function_space = df.FunctionSpace(self.mesh, self.finite_element)
        self.V0 = df.FunctionSpace(self.mesh, "P", 1)
        self.V1 = df.VectorFunctionSpace(self.mesh, "P", 1)
        self.V2 = df.TensorFunctionSpace(self.mesh, "P", 1)

    def solve_TPM_2Phase6Component_MAfLMoCOnCOtCOv_BrainTumour(self):
        """
        test test test
        """

        def output(time):
            nSt_ = df.project(nSt, self.V0)
            nI_ = df.project(nI, self.V0)
            hatrhoSt_ = df.project(hatrhoSt, self.V0)
            hatrhoIt_ = df.project(hatrhoIt, self.V0)
            if self.flag_actConf:
               T_ = df.project(T, self.V2)
            else:
               T_ = df.project(J * T * ufl.inv(F.T), self.V2)

            T_vM_ = df.project(const.calcStress_vonMises(T_), self.V0)

            write_field2xdmf(output_file, u, "u", time)
            write_field2xdmf(output_file, p, "p", time)
            write_field2xdmf(output_file, cIn, "cIn", time)
            write_field2xdmf(output_file, cIt, "cIt", time)
            write_field2xdmf(output_file, cIv, "cIv", time)
            write_field2xdmf(output_file, nSt_, "nSt", time)  # , self.eval_points, self.mesh)
            write_field2xdmf(output_file, nI_, "nI", time)
            write_field2xdmf(output_file, hatrhoSt_, "hatrhoSt", time)
            write_field2xdmf(output_file, hatrhoIt_, "hatrhoIt", time)
            write_field2xdmf(output_file, T_vM_, "vonMises", time)

        prm = df.parameters["form_compiler"]
        prm["quadrature_degree"] = 2

        # **********************************************************************
        # --- get input values -------------------------------------------------
        # **********************************************************************
        # Here only frequently used input variables, such as constants and 
        # functions for the pde should be allocated, others should be allocated
        # by input class. Try not to become to messy
        # -----------------------------------------------------------------------

        # Material Parameters and Parameters that go into weak form
        nS_0S = df.Constant(self.nS_0S)
        rhoShR = df.Constant(self.rhoShR)
        rhoStR = df.Constant(self.rhoStR)
        rhoIR = df.Constant(self.rhoIR)
        molIn = df.Constant(self.molIn)
        molIt = df.Constant(self.molIt)
        molIv = df.Constant(self.molIv)
        kI = df.Constant(self.kI)
        DIn = df.Constant(self.DIn)
        DIt = df.Constant(self.DIt)
        DIv = df.Constant(self.DIv)
        lambdaSh = df.Constant(self.lambdaSh)
        lambdaSt = df.Constant(self.lambdaSt)
        muSh = df.Constant(self.muSh)
        muSt = df.Constant(self.muSt)

        # Time-dependent Parameters
        T_end = self.T_end
        dt = self.dt

        # general
        output_file = self.output_file

        # **********************************************************************
        # --- Calculate solution -----------------------------------------------
        # **********************************************************************

        # Define additional function spaces
        V_DG0 = df.FunctionSpace(self.mesh, "DG", 0)
        V_DG1 = df.FunctionSpace(self.mesh, "DG", 1)

        # Store intern variables
        w_n = df.Function(self.function_space)  # old primaries 
        initTumourTime = df.Function(V_DG0)
        int_nSt = df.Function(V_DG0)
        hatrhoSt = df.Function(V_DG1)
        nSh_0S = nS_0S
        nSt_0S = df.Function(V_DG1)
        hatnSt = df.Function(V_DG1)
        hatrhoI = df.Function(V_DG1)
        hatrhoS = df.Function(V_DG1)
        hatrhoIn = df.Function(V_DG1)
        hatrhoIt = df.Function(V_DG1)
        hatrhoIv = df.Function(V_DG1)
        nI = df.Function(V_DG1)
        dnIdt = df.Function(V_DG1)
        rhoSR = rhoShR

        u_n, p_n, cIn_n, cIt_n, cIv_n = df.split(w_n)

        # Get Ansatz and test functions
        w = df.Function(self.function_space)
        _w = df.TestFunction(self.function_space)
        u, p, cIn, cIt, cIv = df.split(w)
        _u, _p, _cIn, _cIt, _cIv = df.split(_w)

        dx = self.dx

        # Store history values for time integration
        F = kin.calc_defGrad(u)
        J = kin.calc_detDefGrad(u)
        E = kin.calcStrain_GreenLagrange(u)
        E_n = kin.calcStrain_GreenLagrange(u_n)

        # Calculate velocity
        div_v = (1 / dt) * ufl.tr(E - E_n)

        # Calculate storage terms
        dcIndt = (1 / dt) * (cIn - cIn_n)
        dcItdt = (1 / dt) * (cIt - cIt_n)
        dcIvdt = (1 / dt) * (cIv - cIv_n)

        # Calculate volume fractions
        nSh = nSh_0S / J
        nSt = nSt_0S * ufl.exp(hatnSt * dt)
        nS = nSh + nSt
        nI = 1.0 - nS
        dnSdt = hatnSt - nS * div_v
        dnIdt = -dnSdt

        # Calculate seepage-velocity (wtFS)
        nIw_I = -kI * ufl.grad(p)

        # Calculate Diffusion
        nIcInw_In = -DIn * ufl.grad(cIn) + cIn * nIw_I
        nIcItw_It = -DIt * ufl.grad(cIt) + cIt * nIw_I
        nIcIvw_Iv = -DIv * ufl.grad(cIv) + cIv * nIw_I

        # Calculate Stress
        T = const.calcStressExtra_NeoHookean_PiolaKirchhoff2_lin(u, p, lambdaSh, muSh)

        # Define weak forms
        res_BLM = ufl.inner(T, ufl.grad(_u)) * dx
        res_BMI = dnIdt * _p * dx - ufl.inner(nIw_I, ufl.grad(_p)) * dx + nI * div_v * _p * dx - hatrhoI / rhoIR * _p * dx
        res_BCN = nI * dcIndt * _cIn * dx - ufl.inner(nIcInw_In, ufl.grad(_cIn)) * dx + cIn * (
                    div_v - hatrhoSt / rhoStR) * _cIn * dx - hatrhoIn / molIn * _cIn * dx
        res_BCT = nI * dcItdt * _cIt * dx - ufl.inner(nIcItw_It, ufl.grad(_cIt)) * dx + cIt * (
                    div_v - hatrhoSt / rhoStR) * _cIt * dx - hatrhoIt / molIt * _cIt * dx
        res_BCV = nI * dcIvdt * _cIv * dx - ufl.inner(nIcIvw_Iv, ufl.grad(_cIv)) * dx + cIv * (
                    div_v - hatrhoSt / rhoStR) * _cIv * dx - hatrhoIv / molIv * _cIv * dx

        res_tot = res_BLM + res_BMI + res_BCN + res_BCT + res_BCV  # - ip.geom.n_bound

        # Define problem solution
        solver = solv.nonlinvarsolver(res_tot, w, self.d_bound, self.solver_param)

        # Set initial conditions
        w_n.interpolate(self.initial_condition)
        w.interpolate(self.initial_condition)

        # Initialize old step
        u, p, cIn, cIt, cIv = w.split()

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

            # Update for new solution
            u_n, p_n, cIn_n, cIt_n, cIv_n = w_n.split()  # Get old solution
            u, p, cIn, cIt, cIv = w.split()  # Get old solution

            #############################################################
            #                                                           #
            #  Calculate Growth                                         #
            #                                                           #
            #############################################################
            # Send out VEGF if needed
            if self.flag_VEGF:
                hatrhoIv.assign(df.project(self.calc_hatrhoIv(nSt, nI, self.calc_partial_density(cIv * nI, molIv), cIn_n), self.V0))

            # Proliferation
            if self.flag_proliferation:
                hatrhoIt_prolif = self.calc_hatrhoIt_prol(cIn_n, cIt_n)
                hatrhoSt_prolif = self.calc_hatrhoSt_prolif(nSt, cIn_n)
                hatrhoI_prolif = -hatrhoSt_prolif
            else:
                hatrhoIt_prolif = hatrhoSt_prolif = hatrhoI_prolif = 1.0e-60

            # Apoptosis
            if self.flag_apoptose:
                hatrhoIt_apop = self.calc_hatrhoIt_apop(cIt)
                hatrhoSt_apop = self.calc_hatrhoSt_apop(nSt)
                hatrhoI_apop = - hatrhoSt_apop
            else:
                hatrhoIt_apop = hatrhoSt_apop = hatrhoI_apop = 1.0e-60

            # Necrosis
            if self.flag_necrosis:
                hatrhoSt_necros = self.calc_hatrhoSt_nec(nSt, self.calc_partial_density(nSt, self.rhoStR), cIn_n)
                hatrhoI_necros = - hatrhoSt_necros
                hatrhoIt_necros = self.calc_hatrhoIt_nec(cIt_n, cIn_n, nI)
            else:
                hatrhoIt_necros = hatrhoSt_necros = hatrhoI_necros = 1.0e-60

            # Accumulation
            hatrhoSt.assign(df.project(hatrhoSt_apop + hatrhoSt_necros + hatrhoSt_prolif, V_DG1))
            hatrhoIt.assign(df.project(hatrhoIt_apop + hatrhoIt_necros + hatrhoIt_prolif, V_DG1))
            hatrhoI.assign(df.project(hatrhoI_apop + hatrhoI_necros + hatrhoI_prolif, V_DG1))

            # Metabolism
            if self.flag_metabolism:
                hatrhoIn.assign(df.project(self.calc_hatrhoIn(cIn_n, self.calc_mass(cIt, self.molIt, df.CellVolume(self.mesh)), self.calc_mass(nSt, self.rhoStR, df.CellVolume(self.mesh))), self.V0))

            # change in single phase
            initTumourTime.assign(df.project(ufl.conditional(ufl.eq(initTumourTime, 0.0), ufl.conditional(df.ge(cIt_n, self.cIt2nT * 0.8), t, 0.0), initTumourTime), V_DG0))

            # Volume fraction
            nSh = nSh_0S / J
            nSt_0S = ufl.conditional(ufl.gt(initTumourTime, 0.0), (
                        1.0 - self.nS_0S) * self.cIt2nT * molIt / rhoStR, 0.0)
            hatrhoSt_n = ufl.conditional(ufl.ge(initTumourTime, t), hatrhoSt, 0)
            int_nSt.assign(df.project(ufl.conditional(ufl.gt(initTumourTime, 0.0), ufl.conditional(ufl.eq(initTumourTime, t), 1.0, int_nSt + ufl.exp(0.5 * dt * (
                        hatrhoSt_n + hatrhoSt))), 0.0), self.V0))

            nSt = nSt_0S * int_nSt / J
            nS = nSh + nSt
            nI = 1.0 - nS
            dnSdt = hatrhoSt - nS * div_v
            dnIdt = - dnSdt

            # Output solution
            output(t)

            # Calculate current solution
            solver.solve()

            # Update history fields
            w_n.assign(w)