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
        if self.subdomains[cell.index] == 4:
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

    # def value_rank(self):  #    return 1


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
        self.type_nSn = None
        self.type_nSt = None
        self.type_nSv = None
        self.order_u = None
        self.order_p = None
        self.order_nSn = None
        self.order_nSt = None
        self.order_nSv = None
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

    def calc_mass(self, vol_frac, effective_density, ele_volume):
        return self.calc_partial_density(vol_frac, effective_density) * ele_volume

#    def calc_hatrhoIv(self, nSt, nI, rhoIv, cIn):
#        """
#        Calculates production term of VEGF if carcinoma is already there and if nutrients sinks below a threshold value. 
#        Returns density
#        """
#        return ufl.conditional(ufl.ge(nSt, df.DOLFIN_EPS), ufl.conditional(ufl.le(cIn, self.cIn_tresVEGF), self.vegf_max * nI * self.molIv - rhoIv, 0.0), 0.0)
#
#    def calc_hatrhoIn(self, cIn, massIt, massSt):
#        """
#            Calculates proliferation term of mobile cancer cells. First checks if enough nutrients there, then only where concenctration is, growth can happen.
#        """
#        cond1 = ufl.conditional(ufl.gt(massIt, df.DOLFIN_EPS), -self.alpha_In_prolif * massIt, 0.0)
#        cond2 = ufl.conditional(ufl.gt(massSt, df.DOLFIN_EPS), -self.alpha_In_prolif * massSt, 0.0)
#        cond3 = ufl.conditional(ufl.gt(massIt, df.DOLFIN_EPS), -self.alpha_In_survival * massIt, 0.0)
#        cond4 = ufl.conditional(ufl.gt(massSt, df.DOLFIN_EPS), -self.alpha_In_survival * massSt, 0.0)
#        cond5 = ufl.conditional(ufl.ge(cIn, self.cIn_tresGrowthMin), cond3 + cond4, 0.0)
#        return ufl.conditional(ufl.ge(cIn, self.cIn_tresVEGF), cond1 + cond2, cond5)
#
#    def calc_hatrhoIt_prol(self, cIn, cIt):
#        """
#        Calculates proliferation term of mobile cancer cells. First checks if enough nutrients there, then only where concenctration is, growth can happen.
#        """
#        return ufl.conditional(ufl.ge(cIn, self.cIn_tresGrowthMin), ufl.conditional(cIt > 0.0, const.calcGrowth_modVerhulst(cIt, self.molIt, self.kappa_It_prolif, self.cIt2nT, self.dt), 0), 0)
#
#    def calc_hatrhoIt_apop(self, cIt):
#        """
#            Calculates proliferation term of mobile cancer cells. First checks if enough nutrients there, then only where concenctration is, growth can happen.
#        """
#        return ufl.conditional(ufl.gt(cIt, 0.0), -kin.calcGrowth_modVerhulst(cIt, self.molIt, self.kappa_It_apop, self.cIt2nT, self.dt), 0)
#
#    def calc_hatrhoIt_nec(self, cIt, cIn, nI):
#        """
#            Calculates proliferation term of mobile cancer cells. First checks if enough nutrients there, then only where concenctration is, growth can happen.
#        """
#        return ufl.conditional(ufl.gt(cIt, df.DOLFIN_EPS), -kin.calcGrowth_modMonod(cIt * nI, self.mu_It_necros, self.K_It_necros, cIn, self.cIn_tresGrowthMin), 0.0)
#
#    def calc_hatrhoSt_prolif(self, nSt, cIn):
#        """
#        Calculates proliferation term of mobile cancer cells. First checks if enough nutrients there, then only where concenctration is, growth can happen.
#        """
#        return ufl.conditional(ufl.ge(cIn, self.cIn_tresGrowthMin), ufl.conditional(ufl.gt(nSt, 0.0), const.calcGrowth_modVerhulst(nSt, self.rhoStR, self.kappa_T_prolif, self.nT_max, self.dt), 0), 0)
#
#    def calc_hatrhoSt_apop(self, nSt):
#        """
#        Calculates proliferation term of mobile cancer cells. First checks if enough nutrients there, then only where concenctration is, growth can happen.
#        """
#        return ufl.conditional(ufl.gt(nSt, 0.0), -const.calcGrowth_modVerhulst(nSt, self.rhoStR, self.kappa_T_apop, self.nT_max, self.dt), 0)
#
#    def calc_hatrhoSt_nec(self, nSt, rhoSt, cIn):
#        """
#        Calculates proliferation term of mobile cancer cells. First checks if enough nutrients there, then only where concenctration is, growth can happen.
#        """
#        return ufl.conditional(ufl.gt(nSt, df.DOLFIN_EPS), -const.calcGrowth_modMonod(rhoSt, self.mu_T_necros, self.K_T_necros, cIn, self.cIn_tresGrowthMin), 0.0)

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
        self.type_u = input.param.fem.type_u
        self.type_p = input.param.fem.type_p
        self.type_nSn = input.param.fem.type_nSn
        self.type_nSt = input.param.fem.type_nSt
        self.type_nSv = input.param.fem.type_nSv
        self.order_u = input.param.fem.order_u
        self.order_p = input.param.fem.order_p
        self.order_nSn = input.param.fem.order_nSn
        self.order_nSt = input.param.fem.order_nSt
        self.order_nSv = input.param.fem.order_nSv
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
        self.lambdaSt = df.Constant(input.param.mat.lambdaSt)
        self.lambdaSn = df.Constant(input.param.mat.lambdaSt)
        self.muSh = df.Constant(input.param.mat.muSh)
        self.muSt = df.Constant(input.param.mat.muSt)
        self.muSn = df.Constant(input.param.mat.muSn)
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
        element_nSh = df.VectorElement(self.type_nSh, self.mesh.ufl_cell(), self.order_nSh)
        element_nSt = df.VectorElement(self.type_nSt, self.mesh.ufl_cell(), self.order_nSt)
        element_nSn = df.VectorElement(self.type_nSn, self.mesh.ufl_cell(), self.order_nSn)
        element_p = df.FiniteElement(self.type_p, self.mesh.ufl_cell(), self.order_p)
        element_cit = df.FiniteElement(self.type_cit, self.mesh.ufl_cell(), self.order_cit)
        self.finite_element = df.MixedElement(element_u, element_nSh, element_nSt, element_nSn, element_p, element_cit)
        self.function_space = df.FunctionSpace(self.mesh, self.finite_element)
        self.V0 = df.FunctionSpace(self.mesh, "P", 1)
        self.V1 = df.VectorFunctionSpace(self.mesh, "P", 1)
        self.V2 = df.TensorFunctionSpace(self.mesh, "P", 1)

    def solver(self):

        def output(time):
            write_field2output(output_file, cIt, "cIt", time)
            write_field2output(output_file, nSt, "nSt", time)  # , self.eval_points, self.mesh)  # write_field2output(output_file, nI_, "nI", time)  # write_field2output(output_file, hatrhoSt_, "hatrhoSt", time)  # write_field2output(output_file, hatrhoIt_, "hatrhoIt", time)  # write_field2output(output_file, T_vM_, "vonMises", time)

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
        nS_n = nSh_n + nSt_n + nSn_n
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
        C_SE = F_SE.T * F_SE
        J_SE = ufl.det(F_SE)
        B_SE = F_SE * F_SE.T
        E_SE = 0.5 * (C_SE - I)
        dF_Sdt = (F_S - F_Sn) / self.dt
        L_S = dF_Sdt * ufl.inv(F_S)
        D_S = (L_S + L_S.T) / 2.0

        # Calculate velocity
        v = (u - u_n) / dt
        div_v = ufl.inner(D_S, I) 

        # Calculate storage terms
        dcIndt = (1 / dt) * (cIn - cIn_n)
        dcItdt = (1 / dt) * (cIt - cIt_n)
        dcIvdt = (1 / dt) * (cIv - cIv_n)

        # Calculate volume fractions
        nSh = nSh_0S / J_S
        nSt = nSt_0S * ufl.exp(hatnSt * dt)
        nS = nSh + nSt
        nI = 1.0 - nS
        dnSdt = hatnSt - nS * div_v
        dnIdt = -dnSdt

        # Calculate Stress
        lambdaS = (self.lambdaSh * nSh + self.lambdaSt * nSt + self.lambdaSn * nSn) / (nSh + nSt + nSn)
        muS = (self.muSh * nSh + self.muSt * nSt + self.muSn * nSn) / (nSh + nSt + nSn)
        TS_E = (muS * (B_SE - I) + lambdaS * ufl.ln(J_SE) * I) / J_SE
        T = TS_E - p * I
        P_S = J_S * T * ufl.inv(F_S.T)

        # Define weak forms

        #######################################
        # Momentum balance of overall aggregate
        res_LMo1 = ufl.inner(P_S, ufl.grad(_u)) * dx
        res_LMo2 = - J_S * ( (hatrhoS + hatrhoF) + (kappaF * hatrhoF * hatrhoF) / (nF * nF) ) * v * _u * dx
        res_LMo3 = - J_S * hatrhoF * kappaF / nF * ufl.inner(ufl.grad(p), ufl.inv(F_S)) * _u * dx 
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
            nSt_0S = ufl.conditional(ufl.gt(initTumourTime, 0.0), (1.0 - self.nS_0S) * self.cIt2nT * molIt / rhoStR, 0.0)
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
