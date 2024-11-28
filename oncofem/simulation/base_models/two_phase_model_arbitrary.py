"""
Definition of two-phase material according to the Theory of Porous Media. In the fluid constituent multiple components 
can be resolved adaptively.

Class:
    TwoPhaseModel:      Derived from BaseModel. See class description for more information.
"""
import time
from typing import Union
from oncofem.helper.fem_aux import InitialCondition, Solver, MapAverageMaterialProperty
import oncofem.helper.general as gen
from oncofem.problem import Problem
from oncofem.helper.io import write_field2xdmf
import dolfin as df
import ufl
from oncofem.simulation.base_models.base_model import BaseModel

class TwoPhaseArbitraryComponents(BaseModel):
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

        # FEM paramereters
        self.solver_param = None
        self.prim_vars_base = ["u", "p"]
        self.prim_vars = self.prim_vars_base
        self.prim_vars_solid = []
        self.prim_vars_fluid = []
        self.n_prim_vars_base = len(self.prim_vars)
        self.ele_types = ["CG", "CG"]
        self.ele_orders = [2, 1]
        self.tensor_orders = [1, 0]
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
        self.hatnSkappa = []
        self.hatrhoFdelta = []
        self.residuum = None
        self.intern_output = None
        self.solver = None

        # intrinsic material parameters
        self.rhoSkappaR = []
        self.rhoFR = None
        self.gammaFR = None
        self.R = None
        self.Theta = None
        self.healthy_brain_nS = None
        self.alpha_g = None

        # spatial varying material parameters
        self.kF = None
        self.lambdaSkappa = None
        self.muSkappa = None

        # initial conditions
        self.uS_0S = None
        self.p_0S = None
        self.nSkappa_0S = None

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

        # internal quantities
        self.intGrowth = None
        self.intGrowth_n = None

    def set_param(self, ip: Problem) -> None:
        # general parameters
        self.output_file = ip.param.gen.output_file

        # time parameters
        self.T_end = ip.param.time.T_end
        self.output_interval = ip.param.time.output_interval
        self.dt = ip.param.time.dt

        # material parameters base model
        self.rhoFR = df.Constant(ip.param.mat.rhoFR)
        self.gammaFR = df.Constant(ip.param.mat.gammaFR)
        self.R = df.Constant(ip.param.mat.R)
        self.Theta = df.Constant(ip.param.mat.Theta)
        self.healthy_brain_nS = df.Constant(ip.param.mat.healthy_brain_nS)
        self.alpha_g = df.Constant(ip.param.mat.alpha_g)

        # spatial varying material parameters
        self.kF = ip.param.mat.kF

        # Additionals
        if hasattr(ip.param.add, "prim_vars_solid"):
            self.prim_vars_solid = ip.param.add.prim_vars_solid
            self.prim_vars.extend(ip.param.add.prim_vars_solid)
            self.ele_types.extend(ip.param.add.ele_types_solid)
            self.ele_orders.extend(ip.param.add.ele_orders_solid)
            self.tensor_orders.extend(ip.param.add.tensor_orders_solid)
            self.rhoSkappaR = ip.param.add.rhoSkappaR
            self.lambdaSkappa = ip.param.add.lambdaSkappa
            self.muSkappa = ip.param.add.muSkappa
            for idx in range(len(ip.param.add.prim_vars_solid)):
                self.hatnSkappa.append(df.Constant(0.0))

        if hasattr(ip.param.add, "prim_vars_fluid"):
            self.prim_vars_fluid = ip.param.add.prim_vars_fluid
            self.prim_vars.extend(ip.param.add.prim_vars_fluid)
            self.ele_types.extend(ip.param.add.ele_types_fluid)
            self.ele_orders.extend(ip.param.add.ele_orders_fluid)
            self.tensor_orders.extend(ip.param.add.tensor_orders_fluid)
            self.molFdelta = ip.param.add.molFdelta
            self.DFdelta = ip.param.add.DFdelta
            for idx in range(len(ip.param.add.prim_vars_fluid)):
                self.hatrhoFdelta.append(df.Constant(0.0))

        # FEM paramereters
        self.solver_param = ip.param.fem
        # geometry parameters
        self.mesh = ip.geom.mesh
        self.dim = ip.geom.dim
        self.n_bound = ip.geom.n_bound
        self.d_bound = ip.geom.d_bound

    def set_boundaries(self, d_bound, n_bound) -> None:
        self.d_bound = d_bound
        self.n_bound = n_bound    

    def assign_if_function(self, var, index) -> None:
        if type(var) is df.Function:
            df.assign(self.sol.sub(index), var)
            df.assign(self.sol_old.sub(index), var)

    def actualize_prod_terms(self) -> None:
        for idx in range(0, len(self.hatnSkappa)):
            if self.prod_terms[idx] is not None:
                self.hatnSkappa[idx].assign(df.project(self.prod_terms[idx], self.CG1_sca))
            else:
                self.hatnSkappa[idx] = df.Constant(0.0)
        for idx in range(0, len(self.hatrhoFdelta)):
            if self.prod_terms[idx + len(self.hatnSkappa)] is not None:
                self.hatrhoFdelta[idx].assign(df.project(self.prod_terms[idx + len(self.hatnSkappa)], self.CG1_sca))
            else:
                self.hatrhoFdelta[idx] = df.Constant(0.0)

    def set_initial_conditions(self, init, add) -> None:
        """
        Sets initial condition for adaptive system. Can take scalars, distribution from MeshFunctions and Functions.
        """
        # set intern vars
        self.uS_0S = init.uS_0S
        self.p_0S = init.p_0S
        if hasattr(add, "prim_vars_solid"):
            self.nSkappa_0S = add.nSdelta_0S
        if hasattr(add, "prim_vars_fluid"):
            self.cFdelta_0S = add.cFkappa_0S
        # collect for interpolation
        init_set = []
        if type(init.uS_0S) is df.Function:
            init_set = [None] * self.dim
        if type(init.uS_0S) is list:
            for i in range(self.dim):
                init_set.append(gen.check_if_type(init.uS_0S[i], df.Function, None))
        init_set.append(gen.check_if_type(init.p_0S, df.Function, None))
        if self.nSkappa_0S is not None:
            for nSd_0S in self.nSkappa_0S:
                init_set.append(gen.check_if_type(nSd_0S, df.Function, None))
        if self.cFdelta_0S is not None:
            for cFd_0S in self.cFdelta_0S:
                init_set.append(gen.check_if_type(cFd_0S, df.Function, None))
        self.sol.interpolate(InitialCondition(init_set))
        self.sol_old.interpolate(InitialCondition(init_set))

        self.assign_if_function(self.uS_0S, 0)
        self.assign_if_function(self.p_0S, 1)
        if self.nSkappa_0S is not None:
            for idx, nSd_0S in enumerate(self.nSkappa_0S):
                self.assign_if_function(nSd_0S, self.n_prim_vars_base + idx)
        if self.cFdelta_0S is not None:
            for idx, cFd_0S in enumerate(self.cFdelta_0S):
                self.assign_if_function(cFd_0S, self.n_prim_vars_base + len(self.prim_vars_solid) + idx)
        # production terms
        self.actualize_prod_terms()

    def set_function_spaces(self) -> None:
        """
        Sets function space for primary variables u, p, nSdelta, cFgamma and for internal variables
        """
        elements = []
        for idx, e_type in enumerate(self.ele_types):
            if self.tensor_orders[idx] == 0:
                ele = df.FiniteElement(e_type, self.mesh.ufl_cell(), self.ele_orders[idx])
            elif self.tensor_orders[idx] == 1:
                ele = df.VectorElement(e_type, self.mesh.ufl_cell(), self.ele_orders[idx])
            elif self.tensor_orders[idx] == 2:
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

    def set_process_models(self, prod_terms: list) -> None:
        self.prod_terms = prod_terms
        for idx in range(0, len(self.prim_vars_solid)):
            if prod_terms[idx] is not None:
                self.hatnSkappa[idx] = df.Function(self.CG1_sca)
            else:
                self.hatnSkappa[idx] = df.Constant(0.0)
        for idx in range(len(self.prim_vars_solid), len(prod_terms)):
            if prod_terms[idx] is not None:
                self.hatrhoFdelta[idx-len(self.prim_vars_solid)] = df.Function(self.CG1_sca)
            else:
                self.hatrhoFdelta[idx-len(self.prim_vars_solid)] = df.Constant(0.0)

    def output(self, time_step: float) -> None:
        for idx, prim_var in enumerate(self.prim_vars_base):
            write_field2xdmf(self.output_file, self.sol.sub(idx), prim_var, time_step)
        for idx, prim_var in enumerate(self.prim_vars_solid):
            write_field2xdmf(self.output_file, self.sol.sub(idx + self.n_prim_vars_base), prim_var, time_step)  # , function_space=self.CG1_sca)
        for idx, prim_var in enumerate(self.prim_vars_fluid):
            write_field2xdmf(self.output_file, self.sol.sub(idx + self.n_prim_vars_base + len(self.prim_vars_solid)), prim_var, time_step)
        write_field2xdmf(self.output_file, self.intern_output[0], "nS", time_step, function_space=self.CG1_sca)  # , self.eval_points, self.mesh)
        write_field2xdmf(self.output_file, self.intern_output[1], "nF", time_step, function_space=self.CG1_sca)  # , self.eval_points, self.mesh)
        write_field2xdmf(self.output_file, self.intern_output[2], "hatnS", time_step, function_space=self.CG1_sca)  # , self.eval_points, self.mesh)
        write_field2xdmf(self.output_file, self.intern_output[3], "J_Se", time_step, function_space=self.CG1_sca)  # , self.eval_points, self.mesh)
        write_field2xdmf(self.output_file, self.intern_output[4], "J_Sg", time_step, function_space=self.CG1_sca)  # , self.eval_points, self.mesh)
        write_field2xdmf(self.output_file, self.intern_output[5], "P", time_step, function_space=self.CG1_ten)  # , self.eval_points, self.mesh)
        write_field2xdmf(self.output_file, self.hatnSkappa[0], "hatnSh", time_step, function_space=self.CG1_sca)  # , self.eval_points, self.mesh)
        write_field2xdmf(self.output_file, self.hatnSkappa[1], "hatnSt", time_step, function_space=self.CG1_sca)  # , self.eval_points, self.mesh)
        write_field2xdmf(self.output_file, self.hatnSkappa[2], "hatnSn", time_step, function_space=self.CG1_sca)  # , self.eval_points, self.mesh)
        write_field2xdmf(self.output_file, self.hatrhoFdelta[0], "hatrhoFt", time_step, function_space=self.CG1_sca)  # , self.eval_points, self.mesh)
        write_field2xdmf(self.output_file, self.hatrhoFdelta[1], "hatrhoFn", time_step, function_space=self.CG1_sca)  # , self.eval_points, self.mesh)

    def unpack_prim_pvars(self, function_space:df.Function) -> tuple:
        """
        Unpacks primary variables and returns sorted tuple. All prim vars are splitted and gathered into solid and 
        fluid groups.
        """
        u = df.split(function_space) 
        p_solid = []
        p_fluid = []
        for i in range(0, len(self.prim_vars_solid)):
            p_solid.append(u[self.n_prim_vars_base + i])
        for i in range(0, len(self.prim_vars_fluid)):
            p_fluid.append(u[(self.n_prim_vars_base + len(self.prim_vars_solid)) + i])
        return u[0], u[1], p_solid, p_fluid 

    def set_hets_if_needed(self, field: Union[float, MapAverageMaterialProperty]) -> Union[df.Constant, df.Function]:
        if type(field) is float:
            field = df.Constant(field)
        else:
            help_func = field
            field = df.Function(df.FunctionSpace(self.mesh, "DG", 0))
            field.interpolate(help_func)
        return field

    def set_structural_parameters(self) -> None:
        self.kF = self.set_hets_if_needed(self.kF)
        for i in range(len(self.lambdaSkappa)):
            self.lambdaSkappa[i] = self.set_hets_if_needed(self.lambdaSkappa[i])
            self.muSkappa[i] = self.set_hets_if_needed(self.muSkappa[i])
        for i in range(len(self.DFdelta)):
            self.DFdelta[i] = self.set_hets_if_needed(self.DFdelta[i])

    def set_weak_form(self) -> None:
        ##############################################################################
        # Get Ansatz and test functions
        self.sol_old = df.Function(self.function_space)  # old primaries
        self.intGrowth_n = df.Function(self.CG1_sca)
        u, p, nSkappa, cFdelta = self.unpack_prim_pvars(self.ansatz_functions)
        _u, _p, _nSkappa, _cFdelta = self.unpack_prim_pvars(self.test_functions)
        u_n, p_n, nSkappa_n, cFdelta_n = self.unpack_prim_pvars(self.sol_old)
        ##############################################################################
        # Calculate volume fractions
        nS = sum(nSkappa)
        rhoSkappa = [nSd * df.Constant(rhoSdR) for nSd, rhoSdR in zip(nSkappa, self.rhoSkappaR)]
        rhoS = sum([rhoSd / nS for rhoSd in rhoSkappa])
        nF = 1.0 - nS
        ##############################################################################
        # Get growth terms
        hatnS = sum(self.hatnSkappa)
        hatrhoS = sum(hatnSd * df.Constant(rhoSdR) for hatnSd, rhoSdR in zip(self.hatnSkappa, self.rhoSkappaR))
        self.time = df.Constant(0.0)
        ##############################################################################
        # Kinematics with Rodriguez Split
        integral = self.alpha_g * hatnS * (1 - rhoS / self.rhoFR) * self.dt
        cond_time = ufl.conditional(ufl.gt(self.time, 0), integral, 0.0)
        self.intGrowth = self.intGrowth_n + ufl.conditional(ufl.eq(hatrhoS, 0.0), 0.0, cond_time)
        J_Sg = df.exp(self.intGrowth)
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
        div_v = ufl.inner(D_S, I)
        ##############################################################################
        # Calculate Stress
        lambdaS = sum([df.Constant(lambdaSd) * nSd / nS for lambdaSd, nSd in zip(self.lambdaSkappa, nSkappa)])
        muS = sum([df.Constant(muSd) * nSd / nS for muSd, nSd in zip(self.muSkappa, nSkappa)])
        healthy_nF = (1.0 - self.healthy_brain_nS)
        fac_nS = healthy_nF * healthy_nF * (1.0 / healthy_nF - 1.0 / (J_Se - self.healthy_brain_nS))
        TS_E = muS * (B_Se - I) / J_Se + lambdaS * fac_nS * I
        T = TS_E - p * I
        P = J_S * T * ufl.inv(F_S.T)
        ##############################################################################
        # Define weak forms
        kD = df.Constant(self.kF) / df.Constant(self.gammaFR)
        dx = df.Measure("dx", domain=self.mesh)
        ##############################################################################
        # Momentum balance of overall aggregate
        res_LMo1 = ufl.inner(P, ufl.grad(_u)) * dx
        fac_2 = - J_S * hatrhoS * kD / nF
        res_LMo2 = fac_2 * ufl.dot(ufl.dot(ufl.grad(p), ufl.inv(F_S)), _u) * dx
        res_LMo = res_LMo1 + res_LMo2
        ##############################################################################
        # Volume balance of the mixture
        res_VBm1 = J_S * div_v * _p * dx
        res_VBm2u = ufl.dot(ufl.grad(p), ufl.inv(F_S))
        res_VBm2 = J_S * kD * ufl.inner(res_VBm2u, ufl.inv(F_S.T) * ufl.grad(_p)) * dx
        res_VBm3 = - J_S * hatrhoS / rhoS * (1 - rhoS / self.rhoFR) * _p * dx
        res_VBm = res_VBm1 + res_VBm2 + res_VBm3
        ##############################################################################
        # Volume balance of solid bodies
        res_VBkappa = []
        for nSd, nSd_n, hatnSd, _nSd in zip(nSkappa, nSkappa_n, self.hatnSkappa, _nSkappa):
            dnSddt = (nSd - nSd_n) / df.Constant(self.dt)
            res_VB1 = J_S * (dnSddt - hatnSd) * _nSd * dx
            res_VB2 = J_S * nSd * div_v * _nSd * dx
            res_VBkappa.append(res_VB1 + res_VB2)
        ##############################################################################
        # Concentration balance of additionals
        res_CBdelta = []
        for cFd, cFd_n, hatrhoFd, _cFd, DFd, molFd in zip(cFdelta, cFdelta_n, self.hatrhoFdelta, _cFdelta, self.DFdelta, self.molFdelta):
            dcFddt = (cFd - cFd_n) / df.Constant(self.dt)
            dFdelta = DFd / (self.R * self.Theta)
            diffvelo = dFdelta * ufl.dot(ufl.grad(cFd), ufl.inv(C_S))
            seepagevelo = + cFd * kD * ufl.dot(ufl.grad(p), ufl.inv(C_S))
            mass_CBdelta = nF * dcFddt - hatrhoFd / df.Constant(molFd)
            res_CBdelta1 = (mass_CBdelta + cFd * (div_v - hatrhoS / rhoS)) * _cFd
            res_CBdelta2 = ufl.inner(diffvelo, ufl.grad(_cFd)) + ufl.inner(seepagevelo, ufl.grad(_cFd))
            res_CBdelta.append(J_S * (res_CBdelta1 + res_CBdelta2) * dx)
        ##############################################################################
        # Sum up to total residual
        res_tot = res_LMo + res_VBm + sum(res_VBkappa) + sum(res_CBdelta)
        if self.n_bound is not None:
            res_tot += self.n_bound

        self.intern_output = [nS, nF, hatnS, J_Se, J_Sg, P]
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
            self.intGrowth_n.assign(df.project(self.intGrowth, self.CG1_sca))
            self.sol_old.assign(self.sol)
