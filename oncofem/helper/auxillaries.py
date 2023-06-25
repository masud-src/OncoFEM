"""
Definition of auxillary helper functions for the use of fenics are implemented.

Classes:
    BoundingBox:                Defines an area as dolfin subdomain in order to set boundary conditions
    MapAverageMaterialProperty: Averages different values with weights over distributions. Used for the mapping of
                                different material properties.

Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""

import dolfin as df
import ufl
import numpy as np

class BoundingBox(df.SubDomain):
    """
    Defines an area as dolfin subdomain in order to set boundary conditions. Takes set of input bounds and generates
    a cuboid domain.

    methods:
        init:   initialises the bounding box with a mesh (dolfin.mesh) and boundary coordinates (tuples in 2d or 3d)
        inside: defines the inside of the bounding box
    """
    def __init__(self, mesh, x_bounds=None, y_bounds=None, z_bounds=None):
        df.SubDomain.__init__(self)
        self.x_bounds = x_bounds 
        self.y_bounds = y_bounds 
        self.z_bounds = z_bounds
        self.mesh = mesh

    def inside(self, x, on_boundary):
        if self.x_bounds is None:
            x_max = np.max(self.mesh.coordinates()[:, 0])
            x_min = np.min(self.mesh.coordinates()[:, 0])
            x_b = (x_min, x_max)
        else:
            x_b = self.x_bounds
        if self.y_bounds is None:
            y_max = np.max(self.mesh.coordinates()[:, 1])
            y_min = np.min(self.mesh.coordinates()[:, 1])
            y_b = (y_min, y_max)
        else:
            y_b = self.y_bounds
        if self.z_bounds is None:
            z_max = np.max(self.mesh.coordinates()[:, 2])
            z_min = np.min(self.mesh.coordinates()[:, 2])
            z_b = (z_min, z_max)
        else:
            z_b = self.z_bounds
        cond1 = df.between(x[0], x_b)
        cond2 = df.between(x[1], y_b)
        cond3 = df.between(x[2], z_b)
        in_bounding_box = cond1 and cond2 and cond3
        return in_bounding_box and on_boundary

class MapAverageMaterialProperty(df.UserExpression):
    """
    Maps averaged material properties of distributed fields. Used for averaging material parameters of
    grey and white matter and csf. Each of the compartments can have a particular weighting and value.
    Typically used by `set_av_params`. All lists should have the same size.

    methods:
        init:   initialises an averaged mapping of spatially distributed material properties. Can be used directly for
                initial values.
    """
    def __init__(self, values, distributions, weights, **kwargs):
        self.distributions = distributions
        self.values = values
        self.weights = weights
        super().__init__(**kwargs)

    def eval_cell(self, values, x, cell):
        sum = 0
        for i in range(len(self.weights)):
            sum += self.values[i] * self.weights[i] * self.distributions[i][cell.index]
        values[0] = sum

class Solver:
    """
    Definition of solver for non-linear finite-element calculations.
    contains all solver parameters, that can be set.
    
    methods:
        set_non_lin_solver: defines and initialises a non-linear variational problem and set up a solver scheme.
    """
    def __init__(self):
        self.solver_type = "mumps"
        self.maxIter = 20
        self.rel = 1.e-7
        self.abs = 1.e-6
        self.mumps_cntl_1 = 0.05
        self.mumps_icntl_23 = 102400

    def set_non_lin_solver(self, res, x, bcs):
        """
        defines and initialises a non-linear variational problem and set up a solver scheme.

        *Arguments*
            res: residuum of problem, given by variational formulation of set of partial differential equations
            x:   solution vector, in terms of Ax=b
            bcs: dirichlet boundary conditions in form of a list

        *Output*
            gives solver class that can be executed

        *Example*
            solver = solver_object.set_non_lin_solver(residual_Momentum, x, dirichlet_boundaries)
            solver.solve()
        """
        J = df.derivative(res, x)
        problem = df.NonlinearVariationalProblem(res, x, bcs=bcs, J=J)
        solver = df.NonlinearVariationalSolver(problem)
        solver.parameters['newton_solver']['maximum_iterations'] = self.maxIter
        solver.parameters['newton_solver']['relative_tolerance'] = self.rel
        solver.parameters['newton_solver']['absolute_tolerance'] = self.abs
        solver.parameters['newton_solver']['linear_solver'] = self.solver_type
        if self.solver_type == "mumps":
            df.PETScOptions.set("-mat_mumps_cntl_1", self.mumps_cntl_1)  # relative pivoting threshold
            df.PETScOptions.set("-mat_mumps_icntl_23", self.mumps_icntl_23)  # max size of the working memory (MB) that can allocate per processor
        return solver

def set_av_params(params, distributions, weights):
    """
        Maps averaged material properties of distributed fields. Typically used with lists of parameters and particular
        distributions and weights. All lists should have the same size.

        *Arguments*:
            params: List of floats
            distributions: List of distributions hold in a dolfin Meshfunction
            weights: List of floats
        *Example*:
            diffusion_drug = set_av_params([1e-7, 1e-8], [white_matter, grey_matter], [1, 1]) 
    """
    return MapAverageMaterialProperty(params, distributions, weights)

def calcStress_vonMises(T):
    """
    calculates scalar von Mises stress
    *Arguments:*
        T: Stress tensor (2D/3D)
    *Example:*
        calcStress_vonMises(T)
    """
    sig_x = T[0, 0]
    sig2_x = sig_x * sig_x
    sig_y = T[1, 1]
    sig2_y = sig_y * sig_y
    tau_xy = T[1, 0] * T[1, 0]
    if ufl.shape(T)[0] == 2:
        return ufl.sqrt(sig2_x + sig2_y - sig_x * sig_y + 3.0 * tau_xy)
    elif ufl.shape(T)[0] == 3:
        sig_z = T[2, 2]
        sig2_z = sig_z * sig_z
        tau_xz = T[0, 2] * T[0, 2]
        tau_yz = T[1, 2] * T[1, 2]
        return ufl.sqrt(sig2_x + sig2_y + sig2_z - sig_x * sig_y - sig_x * sig_z - sig_y * sig_z + 3.0 * (
                    tau_xy + tau_xz + tau_yz))

def meshfunction_2_function(mf: df.MeshFunction, fs: df.FunctionSpace):
    """
    maps meshfunction to functionspace. Only works with constant meshfunction space and linear functionspace

    *Arguments*:
        mf: dolfin Meshfunction
        fs: dolfin Functionspace
    *Example*:
        pressure = meshfunction_2_function(pressure_mesh_function, pressure_function_space)
    """
    v2d = dolfin.vertex_to_dof_map(fs)
    u = dolfin.Function(fs)
    u.vector()[v2d] = mf.array()
    return u
