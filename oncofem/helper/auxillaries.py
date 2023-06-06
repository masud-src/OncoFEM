"""
Definition of auxillary helper functions for the use of fenics are implemented.

Classes:
    BoundingBox:                Defines an area as dolfin subdomain in order to set boundary conditions
    MapAverageMaterialProperty: Averages different values with weights over distributions. Used for the mapping of
                                different material properties.

Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""

import dolfin
import ufl
import numpy as np

class BoundingBox(dolfin.SubDomain):
    """
    Defines an area as dolfin subdomain in order to set boundary conditions. Takes set of input bounds and generates
    a cuboid domain.

    methods:
        init:   initialises the bounding box with a mesh (dolfin.mesh) and boundary coordinates (tuples in 2d or 3d)
        inside: defines the inside of the bounding box
    """
    def __init__(self, mesh, x_bounds=None, y_bounds=None, z_bounds=None):
        dolfin.SubDomain.__init__(self)
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
        cond1 = dolfin.between(x[0], x_b)
        cond2 = dolfin.between(x[1], y_b)
        cond3 = dolfin.between(x[2], z_b)
        in_bounding_box = cond1 and cond2 and cond3
        return in_bounding_box and on_boundary

class MapAverageMaterialProperty(dolfin.UserExpression):
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

def assign_load_curve(t, f_magnitude, dt_1, dt_2, q_max):
    """
    assigns a load curve with a first linear progression, that turns into constant at time step dt_1. 
    The load vanishes at time dt_2

    *Arguments*:
        t: actual time step
        f_magnitude: value that shall increase
        dt_1: time, when linear function becomes constant
        dt_2: time, when constant function becomes zero
        q_max: Maximum of increasing value

    *Example*:
        assign_load_curve(t, bF_mag_ds3, 10, 20, 100)
    """
    if t < dt_1:
        f_magnitude.assign(t/dt_1*q_max)
    elif t < dt_2:
        f_magnitude.assign(q_max)
    else:
        f_magnitude.assign(0.0)

def rotation_matrix(alpha: float, dim: int, beta=None, gamma=None):
    """
    rotates a two or three dimensional matrix

    *Arguments:*
        alpha: float: angle in degree
        dim: integer: 2 or 3
        beta: float: angle in degree
        gamma: float: angle in degree

    *Example:*
        matrix = rotate_matrix(90.0, 3, 0.0, 10.5)
        matrix = rotate_matrix(90.0, 2)    
    """
    a = alpha; b = beta; g = gamma
    if dim == 2:
        return ufl.as_matrix([[ufl.cos(a), -ufl.sin(a)], [ufl.sin(a), ufl.cos(a)]])
    if dim == 3:
        return ufl.as_matrix([[ufl.cos(a)*ufl.cos(b), ufl.cos(a)*ufl.sin(b)*ufl.sin(g) - ufl.sin(a)*ufl.cos(g), ufl.cos(a)*ufl.sin(b)*ufl.cos(g) + ufl.sin(a)*ufl.sin(g)],
                   [ufl.sin(a)*ufl.cos(b), ufl.sin(a)*ufl.sin(b)*ufl.sin(g) + ufl.cos(a)*ufl.cos(g), ufl.sin(a)*ufl.sin(b)*ufl.cos(g) - ufl.cos(a)*ufl.sin(g)],
                   [-ufl.sin(a), ufl.cos(b)*ufl.cos(g), ufl.cos(b)*ufl.cos(g)]])

def square_norm(field: dolfin.Function):
    """
    Applies the square norm onto a dolfin function.

    *Arguments:*
        field: dolfin Function

    *Example:*
        normed_function = square_norm(field)
    """
    return ufl.sqrt(sum([x*x for x in field.split()]))

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

def meshfunction_2_function(mf: dolfin.MeshFunction, fs: dolfin.FunctionSpace):
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
