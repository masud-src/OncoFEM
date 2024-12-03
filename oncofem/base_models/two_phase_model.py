"""
Definition of two-phase material according to the Theory of Porous Media. In the fluid constituent multiple components 
can be resolved adaptively.

Class:
    TwoPhaseModel:      Derived from BaseModel. See class description for more information.
"""
import time
from typing import Union
from oncofem.utils.fem import InitialCondition, Solver, MapAverageMaterialProperty
import oncofem.utils.general as gen
from oncofem.utils.structure import Problem
from oncofem.utils.io import write_field2xdmf
import dolfin as df
import ufl
from oncofem.base_models.base_model import BaseModel

class TwoPhaseModel(BaseModel):
    """
    The two phase model implements a two phase material in the continuum-mechanical framework of the Theory of Porous
    Media. The material is split into a fluid and solid part, wherein the fluid part multiple components can be
    resolved. The user can either set free defined functions, constants or load xdmf input files to set initial 
    conditions. In order to have time dependent production terms or to couple the production terms to other software
    the production terms will be actualised in every time step.  

    *Methods:*
        set_boundaries:         Sets surface boundaries, e. g. Dirichlet and Neumann boundaries.
        assign_if_function:     Helper function, that assigns values to the solution field and old solution field, if a
                                function is given. Used for adaptive initial conditions.
        actualize_prod_terms:   Actualises production terms in each time step.
        set_initial_conditions: Sets initial condition for adaptive system. Can take scalars, distribution from 
                                MeshFunctions and Functions.
        set_function_spaces:    Sets function space for primary variables u, p, nS, cFdelta and for internal .
        set_param:              Sets parameter needed for model class from given problem.
        set_process_models:     Sets the chosen bio-chemical model set-up on the microscale.
        output:                 Defines the way the output shall be created and what shall be exported.
        unpack_prim_pvars:      Unpacks primary variables and returns tuple.
        set_hets_if_needed:     Sets the heterogenities, if a distribution is given
        set_heterogeneities:    Set heterogenities on the domain.     
        set_weak_form:          Sets the weak form of the system of partial differential equations.  
        set_solver:             Sets up the numerical solver method to solve the weak form.                
        solve:                  Method for solving the particular model within one time interval.
    """
    def __init__(self):
        super().__init__()
        # general info
        self.output_file = None
        self.flag_defSplit = False

        # FEM paramereters
        self.solver_param = None
        self.prim_vars_list = ["u", "p", "nS"]
        self.n_init_prim_vars = len(self.prim_vars_list)
        self.tensor_order = [1, 0, 0]
        self.ele_types = ["CG", "CG", "CG"]
        self.ele_orders = [2, 1, 1]
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
        self.hatnS = None
        self.hatrhoFkappa = []
        self.residuum = None
        self.intern_output = None
        self.solver = None

        # intrinsic material parameters
        self.rhoSR = None
        self.rhoFR = None
        self.gammaFR = None
        self.R = None
        self.Theta = None
        self.healthy_brain_nS = None

        # spatial varying material parameters
        self.kF = None
        self.lambdaS = None
        self.muS = None
        self.DFt = None

        # initial conditions
        self.uS_0S = None
        self.p_0S = None
        self.nS_0S = None

        # additional concentrations
        self.cFkappa = None
        self.molFkappa = None
        self.DFkappa = None
        self.cFkappa_0S = None

        # time parameters
        self.time = None
        self.T_end = None
        self.output_interval = None
        self.dt = None

        # internal quantities
        self.intGrowth = None
        self.intGrowth_n = None

    def set_boundaries(self, d_bound, n_bound) -> None:
        self.d_bound = d_bound
        self.n_bound = n_bound    

    def assign_if_function(self, var, index) -> None:
        if type(var) is df.Function:
            df.assign(self.sol.sub(index), var)
            df.assign(self.sol_old.sub(index), var)

    def actualize_prod_terms(self) -> None:
        if self.prod_terms[0] is not None:
            self.hatnS.assign(df.project(self.prod_terms[0], self.CG1_sca))
        else:
            self.hatnS = df.Constant(0.0)

        for idx in range(1, len(self.prod_terms)):
            if self.prod_terms[idx] is not None:
                self.hatrhoFkappa[idx-1].assign(df.project(self.prod_terms[idx], self.CG1_sca))
            else:
                self.hatrhoFkappa[idx-1] = df.Constant(0.0)

    def set_initial_conditions(self, init, add) -> None:
        """
        Sets initial condition for adaptive system. Can take scalars, distribution from MeshFunctions and Functions.
        """
        # set intern vars
        self.uS_0S = init.uS_0S
        self.p_0S = init.p_0S
        self.nS_0S = init.nS_0S
        if hasattr(add, "prim_vars"):
            self.cFkappa_0S = add.cFkappa_0S
        # collect for interpolation
        init_set = []
        if type(init.uS_0S) is df.Function:
            init_set = [None] * self.dim
        if type(init.uS_0S) is list:
            for i in range(self.dim):
                init_set.append(gen.check_if_type(init.uS_0S[i], df.Function, None))
        init_set.append(gen.check_if_type(init.p_0S, df.Function, None))
        init_set.append(gen.check_if_type(init.nS_0S, df.Function, None))
        if self.cFkappa_0S is not None:
            for cFd_0S in self.cFkappa_0S:
                init_set.append(gen.check_if_type(cFd_0S, df.Function, None))
        self.sol.interpolate(InitialCondition(init_set))
        self.sol_old.interpolate(InitialCondition(init_set))

        self.assign_if_function(self.uS_0S, 0)
        self.assign_if_function(self.p_0S, 1)
        self.assign_if_function(self.nS_0S, 2)
        if self.cFkappa_0S is not None:
            for idx, cFd_0S in enumerate(self.cFkappa_0S):
                self.assign_if_function(cFd_0S, 3+idx)
        # production terms
        self.actualize_prod_terms()

    def set_function_spaces(self) -> None:
        """
        Sets function space for primary variables u, p, nS, cFdelta and for internal variables
        """
        elements = []
        for idx, e_type in enumerate(self.ele_types):
            if self.tensor_order[idx] == 0:
                ele = df.FiniteElement(e_type, self.mesh.ufl_cell(), self.ele_orders[idx])
            elif self.tensor_order[idx] == 1:
                ele = df.VectorElement(e_type, self.mesh.ufl_cell(), self.ele_orders[idx])
            elif self.tensor_order[idx] == 2:
                ele = df.TensorElement(e_type, self.mesh.ufl_cell(), self.ele_orders[idx])
            elements.append(ele)
        self.finite_element = ufl.MixedElement(elements)
        self.function_space = df.FunctionSpace(self.mesh, self.finite_element)
        self.DG0 = df.FunctionSpace(self.mesh, "DG", 0)
        self.CG1_sca = df.FunctionSpace(self.mesh, "CG", 1)
        self.CG1_vec = df.VectorFunctionSpace(self.mesh, "CG", 1)
        self.CG1_ten = df.TensorFunctionSpace(self.mesh, "CG", 1)
        self.ansatz_functions = df.Function(self.function_space)
        self.test_functions = df.TestFunction(self.function_space)

    def set_param(self, ip:Problem) -> None:
        # general parameters
        self.output_file = ip.param.gen.output_file
        self.flag_defSplit = ip.param.gen.flag_defSplit

        # time parameters
        self.T_end = ip.param.time.T_end
        self.output_interval = ip.param.time.output_interval
        self.dt = ip.param.time.dt

        # material parameters base model
        self.rhoSR = df.Constant(ip.param.mat.rhoSR)
        self.rhoFR = df.Constant(ip.param.mat.rhoFR)
        self.gammaFR = df.Constant(ip.param.mat.gammaFR)
        self.R = df.Constant(ip.param.mat.R)
        self.Theta = df.Constant(ip.param.mat.Theta)
        self.healthy_brain_nS = df.Constant(ip.param.mat.healthy_brain_nS)

        # spatial varying material parameters
        self.kF = ip.param.mat.kF
        self.lambdaS = ip.param.mat.lambdaS
        self.muS = ip.param.mat.muS

        # FEM paramereters and additionals
        self.solver_param = ip.param.fem
        if hasattr(ip.param.add, "prim_vars"):
            self.prim_vars_list.extend(ip.param.add.prim_vars)
            self.ele_types.extend(ip.param.add.ele_types)
            self.ele_orders.extend(ip.param.add.ele_orders)
            self.tensor_order.extend(ip.param.add.tensor_orders)
            self.molFkappa = ip.param.add.molFkappa
            self.DFkappa = ip.param.add.DFkappa
            for idx in range(len(ip.param.add.prim_vars)):
                self.hatrhoFkappa.append(df.Constant(0.0))

        # geometry parameters
        self.mesh = ip.geom.mesh
        self.dim = ip.geom.dim
        self.n_bound = ip.geom.n_bound
        self.d_bound = ip.geom.d_bound

    def set_process_models(self, prod_terms:list) -> None:
        self.prod_terms = prod_terms
        self.hatnS = df.Function(self.CG1_sca)
        for idx in range(1, len(prod_terms)):
            if prod_terms[idx] is not None:
                self.hatrhoFkappa[idx-1] = df.Function(self.CG1_sca)
            else:
                self.hatrhoFkappa[idx-1] = df.Constant(0.0)

    def output(self, time_step:float) -> None:
        for idx, prim_var in enumerate(self.prim_vars_list):
            write_field2xdmf(self.output_file, self.sol.sub(idx), prim_var, time_step)
        write_field2xdmf(self.output_file, self.intern_output[0], "hatcFt", time_step, function_space=self.CG1_sca)  # , self.eval_points, self.mesh)
        write_field2xdmf(self.output_file, self.intern_output[1], "hatnS", time_step, function_space=self.CG1_sca)  # , self.eval_points, self.mesh)

    def unpack_prim_pvars(self, function_space:df.FunctionSpace) -> tuple:
        """unpacks primary variables and returns tuple"""
        u = df.split(function_space)
        p = []
        for i in range(self.n_init_prim_vars, len(u)):
            p.append(u[i])
        return u[0], u[1], u[2], p 

    def set_hets_if_needed(self, field:Union[float, MapAverageMaterialProperty]) -> Union[df.Constant, df.Function]:
        if type(field) is float:
            field = df.Constant(field)
        else:
            help_func = field
            field = df.Function(df.FunctionSpace(self.mesh, "DG", 0))
            field.interpolate(help_func)
        return field

    def set_structural_parameters(self) -> None:
        self.kF = self.set_hets_if_needed(self.kF)
        self.lambdaS = self.set_hets_if_needed(self.lambdaS)
        self.muS = self.set_hets_if_needed(self.muS)
        for i in range(len(self.DFkappa)):
            self.DFkappa[i] = self.set_hets_if_needed(self.DFkappa[i])

    def set_weak_form(self) -> None:
        ##############################################################################
        # Get Ansatz and test functions
        self.sol_old = df.Function(self.function_space)  # old primaries
        u, p, nS, cFkappa = self.unpack_prim_pvars(self.ansatz_functions)
        _u, _p, _nS, _cFkappa = self.unpack_prim_pvars(self.test_functions)
        u_n, p_n, nS_n, cFkappa_n = self.unpack_prim_pvars(self.sol_old)
        ##############################################################################
        # Calculate volume fractions
        nF = 1.0 - nS
        ##############################################################################
        # Get growth terms
        hatnS = self.hatnS
        hatrhoS = hatnS * df.Constant(self.rhoSR) 
        hatrhoF = - hatrhoS
        self.intGrowth_n = df.Function(self.CG1_sca)
        self.time = df.Constant(0.0)
        ##############################################################################
        # Kinematics with Rodriguez Split
        if self.flag_defSplit:
            growth = hatnS / nS_n * self.dt
            growth_increment = ufl.conditional(ufl.gt(self.time, 0), growth, 0.0)
            self.intGrowth = self.intGrowth_n + growth_increment
            J_Sg = df.exp(self.intGrowth)
        else:
            J_Sg = 1.0

        I = ufl.Identity(len(u))
        F_Sg = J_Sg ** (1 / len(u)) * I
        F_S = I + ufl.grad(u)
        F_Sn = I + ufl.grad(u_n)
        dF_Sdt = (F_S - F_Sn) / df.Constant(self.dt)
        L_S = dF_Sdt * ufl.inv(F_S)
        D_S = (L_S + L_S.T) / 2.0
        C_S = F_S.T * F_S
        J_S = ufl.det(F_S)
        F_Se = F_S * ufl.inv(F_Sg)
        J_Se = ufl.det(F_Se)
        B_Se = F_Se * F_Se.T
        ##############################################################################
        # Calculate velocities
        v = (u - u_n) / df.Constant(self.dt)
        div_v = ufl.inner(D_S, I)
        dnSdt = (nS - nS_n) / df.Constant(self.dt)
        dcFkappadt = []
        for i, cFk in enumerate(cFkappa):
            dcFkappadt.append((cFk - cFkappa_n[i]) / df.Constant(self.dt))
        ##############################################################################
        # Calculate Stress
        lambdaS = df.Constant(self.lambdaS)
        muS = df.Constant(self.muS)
        healthy_nF = (1.0 - self.healthy_brain_nS)
        fac_nS = healthy_nF * healthy_nF * (1.0 / healthy_nF - 1.0 / (J_Se - self.healthy_brain_nS))
        TS_E = muS * (B_Se - I) / J_Se + lambdaS * fac_nS * I
        T = TS_E - nS * p * I
        P = J_S * T * ufl.inv(F_S.T)
        ##############################################################################
        # Define weak forms
        kD = df.Constant(self.kF) / df.Constant(self.gammaFR)
        dx = df.Measure("dx", domain=self.mesh)
        ##############################################################################
        # Momentum balance of overall aggregate
        res_LMo1 = ufl.inner(P, ufl.grad(_u)) * dx
        fac_1 = + J_S * kD / (nF*nF) * hatrhoF * hatrhoF
        res_LMo2 = fac_1 * ufl.dot(ufl.dot(v, ufl.inv(F_S)), _u) * dx
        fac_2 = - J_S * kD / nF
        res_LMo3 = fac_2 * ufl.dot(ufl.dot(ufl.grad(p), ufl.inv(F_S)), _u) * dx
        res_LMo = res_LMo1 + res_LMo2 + res_LMo3
        ##############################################################################
        # Volume balance of the mixture
        res_VBm1 = J_S * div_v * _p * dx 
        res_VBm2 = - J_S * (hatrhoS / self.rhoSR + hatrhoF / self.rhoFR) * _p * dx
        velo_part = hatrhoF / nF * ufl.dot(v, ufl.inv(F_S.T))
        res_VBm31 = ufl.dot(ufl.grad(p), ufl.inv(C_S)) + velo_part 
        res_VBm3 = J_S * kD * ufl.inner(res_VBm31, ufl.grad(_p)) * dx
        res_VBm = res_VBm1 + res_VBm2 + res_VBm3
        ##############################################################################
        # Volume balance of solid body
        res_VB1 = J_S * (dnSdt - hatnS) * _nS * dx
        res_VB2 = J_S * nS * div_v * _nS * dx
        res_VB = res_VB1 + res_VB2
        ##############################################################################
        # Concentration balance of additionals
        res_CBkappa = []
        for i, cFk in enumerate(cFkappa):
            dFkappa = self.DFkappa[i] / (self.R * self.Theta)
            prodvelo = self.hatrhoFkappa[i] / nF * ufl.dot(v, ufl.inv(F_S.T))
            diffvelo = dFkappa * (ufl.dot(ufl.grad(cFk), ufl.inv(C_S)) + prodvelo)
            seepagevelo = - cFk * kD * (ufl.dot(ufl.grad(p), ufl.inv(C_S)) - prodvelo)
            mass_CBkappa = nF * dcFkappadt[i] - self.hatrhoFkappa[i] / df.Constant(self.molFkappa[i])
            res_CBkappa1 = mass_CBkappa * _cFkappa[i]
            res_CBkappa2 = cFk * (div_v - hatrhoS / self.rhoSR) * _cFkappa[i]
            res_CBkappa3 = ufl.inner(diffvelo, ufl.grad(_cFkappa[i]))
            res_CBkappa4 = ufl.inner(seepagevelo, ufl.grad(_cFkappa[i]))
            res_CBkappa.append(J_S * (res_CBkappa1 + res_CBkappa2 + res_CBkappa3 + res_CBkappa4) * dx)
        ##############################################################################
        # Sum up to total residual
        res_tot = res_LMo + res_VBm + res_VB
        for res_CBk in res_CBkappa:
            res_tot += res_CBk
        if self.n_bound is not None:
            res_tot += self.n_bound

        self.intern_output = [self.hatrhoFkappa[0], hatnS]
        self.residuum = res_tot

    def set_solver(self) -> None:
        prm = df.parameters["form_compiler"]
        prm["quadrature_degree"] = 2  # Make sure quadrature_degree stays at 2
        self.sol = self.ansatz_functions
        solver = Solver()
        solver.solver_type = self.solver_param.solver_type
        solver.abs = self.solver_param.abs
        solver.rel = self.solver_param.rel
        solver.maxIter = self.solver_param.maxIter
        self.solver = solver.set_non_lin_solver(self.residuum, self.sol, self.d_bound)

    def solve(self) -> None:
        # Initialize  and time loop
        t = 0.0
        out_count = 0.0
        time_flag = True
        self.output(t)
        print("Initial step is written")
        while t < self.T_end:
            # Increment solution time
            t = t + self.dt
            self.time.assign(t)
            out_count += self.dt
            # Calculate current solution
            if time_flag:
                timer_start = time.time()
                time_flag = False
            n_iter, converged = self.solver.solve()
            # actualize prod terms
            self.actualize_prod_terms()
            # Output solution
            if out_count >= self.output_interval:
                timer_end = time.time()
                time_flag = True
                print("Time: {}".format(t), "  ", "Converged in steps: {}".format(n_iter), " ", 
                      "Calculation time: {:.2f}".format(timer_end - timer_start), 
                      "finish_meter: {:.2f}".format(t/self.T_end))
                out_count = 0.0
                self.output(t)
            # Update history fields
            if self.flag_defSplit:
                self.intGrowth_n.assign(df.project(self.intGrowth, self.CG1_sca))
            self.sol_old.assign(self.sol)
