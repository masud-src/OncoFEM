"""
Definition of auxillary helper functions for the use of the finite element method via fenics are implemented.

Classes:
    InitialDistribution:        Defines an initial distribution of a field.
    InitialCondition:           Defines an initial distribution of a primary field.
    BoundingBox:                Defines an area as dolfin subdomain in order to set boundary conditions
    MapAverageMaterialProperty: Averages different values with weights over distributions. Used for the mapping of
                                different material properties.
    Solver:                     Definition of solver for non-linear finite-element calculations.

Functions:
    mark_facet:                 Marks the facets made by bounding boxes with integers from 1. 
    set_av_params:              Sets averaged material parameters, according to specific distributions and weights
    calStress_vonMises:         Calculates von Mises stress
    meshfunction_2_function:    Maps a meshfunction to a function. Only works with constant meshfunction space and 
                                linear functionspace
"""
from typing import Union
import dolfin as df
import ufl
import numpy as np
from abc import ABC

class InitialDistribution(df.UserExpression, ABC):
    """
    Defines an initial distribution of a scalar field.

    methods:
        init:       initialises and sets the value
        eval_cell:  evaluates the value at particular cells
    """
    def __init__(self, value, **kwargs):
        self.value = value
        super().__init__(**kwargs)

    def eval_cell(self, values, x, cell):
        values[0] = self.value[cell.index]

    def value_shape(self):
        return self.size,


class InitialCondition(df.UserExpression, ABC):
    """
    Defines an initial distribution of a primary field.

    methods:
        init:               initialises and sets the initial value set
        case_distinction:   sets zero if condition is None
        eval_cell:          evaluates the value at particular cells
        value_shape:        returns the value shape
    """
    def __init__(self, init_set,  **kwargs):
        self.init_set = self.case_distinction(init_set)
        self.size = len(init_set)
        super().__init__(**kwargs)

    def case_distinction(self, init_cond: list):
        for idx, cond in enumerate(init_cond):
            if cond is None:
                init_cond[idx] = 0.0
        return init_cond

    def eval_cell(self, values, x, cell):
        for idx,val in enumerate(values):
            if type(self.init_set[idx]) is float:
                values[idx] = self.init_set[idx]
            else:
                values[idx] = self.init_set[idx][cell.index]

    def value_shape(self):
        return self.size,

class BoundingBox(df.SubDomain):
    """
    Defines an area as dolfin subdomain in order to set boundary conditions. Takes set of input bounds and generates
    a cuboid domain.

    methods:
        init:   initialises the bounding box with a mesh (dolfin.mesh) and boundary coordinates (tuples in 2d or 3d)
        inside: defines the inside of the bounding box
    """
    def __init__(self, mesh, bounds=None):
        df.SubDomain.__init__(self)
        self.bounds = bounds
        self.mesh = mesh

    def inside(self, x, on_boundary) -> bool:
        """
        Checks if cells on material point x are inside boundary area.
        """
        bound_coord = []
        cond = []
        for i, bound in enumerate(self.bounds):
            if bound is None:
                max = np.max(self.mesh.coordinates()[:, 0])
                min = np.min(self.mesh.coordinates()[:, 0])
                bound_coord.append((min, max))
            else:
                bound_coord.append(bound)
            cond.append(df.between(x[i], bound_coord[i]))

        return all(cond) and on_boundary

class MapAverageMaterialProperty(df.UserExpression, ABC):
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
        values[0] = df.Constant(sum)

    def value_shape(self):
        return ()

class Solver:
    """
    Definition of solver for non-linear finite-element calculations.
    Contains all solver parameters, that can be set.

    methods:
        set_non_lin_solver: defines and initialises a non-linear variational problem and set up a solver scheme.
    """
    def __init__(self):
        self.solver_type = "mumps"
        self.maxIter = 20
        self.rel = 1.e-7
        self.abs = 1.e-6
        self.mumps_cntl_1 = 0.05            # relative pivoting threshold
        self.mumps_icntl_23 = 102400        # max size of the working memory (MB) that can allocate per processor

    def set_non_lin_solver(self, res: df.Form, x: df.Function, bcs: list) -> df.NonlinearVariationalSolver:
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
            df.PETScOptions.set("-mat_mumps_cntl_1", self.mumps_cntl_1)  
            df.PETScOptions.set("-mat_mumps_icntl_23", self.mumps_icntl_23)  
        return solver

def mark_facet(mesh: df.Mesh, bounding_boxes: list, directory=None) -> tuple[df.MeshFunction, df.MeshFunction]:
    """
    Marks the facets made by bounding boxes. 

    *Arguments*:
        mesh:           dolfin mesh entity that will be marked
        bounding_boxes: List of bounding boxes
        directory:      String, optional output directory, where "surface.xdmf" will be saved

    *Example*:
        mf_domain, mf_facet = mark_facet(mesh, [bounding_box_brainstem, bounding_box_cerebellum])
    """
    mf_domain = df.MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
    mf_facet = df.MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    for i, bounding_box in enumerate(bounding_boxes):
        bounding_box.mark(mf_facet, i+1)
    if directory is not None:
        surf_xdmf_file = directory + "surface.xdmf"
        df.XDMFFile.write(df.XDMFFile(surf_xdmf_file), mf_facet)
    return mf_domain, mf_facet

def set_av_params(params: list[float], distributions: list[df.MeshFunction], weights: list[float]) -> MapAverageMaterialProperty:
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

def calcStress_vonMises(stress) -> Union[float, complex]:
    """
    calculates scalar von Mises stress

    *Arguments:*
        T: Stress tensor (2D/3D)
    *Example:*
        calcStress_vonMises(T)
    """
    sig_x = stress[0, 0]
    sig2_x = sig_x * sig_x
    sig_y = stress[1, 1]
    sig2_y = sig_y * sig_y
    tau_xy = stress[1, 0] * stress[1, 0]
    if ufl.shape(stress)[0] == 2:
        return ufl.sqrt(sig2_x + sig2_y - sig_x * sig_y + 3.0 * tau_xy)
    elif ufl.shape(stress)[0] == 3:
        sig_z = stress[2, 2]
        sig2_z = sig_z * sig_z
        tau_xz = stress[0, 2] * stress[0, 2]
        tau_yz = stress[1, 2] * stress[1, 2]
        sig_xy = sig_x * sig_y
        sig_xz = sig_x * sig_z
        sig_yz = sig_y * sig_z
        return ufl.sqrt(sig2_x + sig2_y + sig2_z - sig_xy - sig_xz - sig_yz + 3.0 * (tau_xy + tau_xz + tau_yz))

def meshfunction_2_function(mf: df.MeshFunction, fs: df.FunctionSpace) -> df.Function:
    """
    maps meshfunction to functionspace. Only works with constant meshfunction space and linear functionspace

    *Example*:
        pressure = meshfunction_2_function(pressure_mesh_function, pressure_function_space)

    :param mf: dolfin Meshfunction
    :param fs: dolfin Functionspace

    :return: dolfin Function
    """
    v2d = df.vertex_to_dof_map(fs)
    u = df.Function(fs)
    u.vector()[v2d] = mf.array()
    return u
