"""
# **************************************************************************#
#                                                                           #
# === Auxillaries ==========================================================#
#                                                                           #
# **************************************************************************#
# Definition of auxillary helper functions
#
# Co-author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
# Co-author: Maximilian Brodbeck <maximilian.brodbeck@isd.uni-stuttgart.de>
#
# --------------------------------------------------------------------------#
"""

import dolfin
import ufl

def assign_load_curve(t, bF_magnitude, dt_1, dt_2, q_max):
    """
    assigns a load curve with a first linear progression, that turns into constant at time step dt_1. 
    The load vanishes at time dt_2

    *Arguments*
        t: actual time step
        bF_magnitude: value that shall increase
        dt_1: time, when linear function becomes constant
        dt_2: time, when constant function becomes zero
        q_max: Maximum of increasing value

    *Example*
        assign_load_curve(t, bF_mag_ds3, 10, 20, 100)
    """
    if t < dt_1:
        bF_magnitude.assign(t/dt_1*q_max)
    elif t < dt_2:
        bF_magnitude.assign(q_max)
    else:
        bF_magnitude.assign(0.0)

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
                   [ufl.sin(a)*ufl.cos(b)       , ufl.sin(a)*ufl.sin(b)*ufl.sin(g) + ufl.cos(a)*ufl.cos(g), ufl.sin(a)*ufl.sin(b)*ufl.cos(g) - ufl.cos(a)*ufl.sin(g)],
                   [-ufl.sin(a)             , ufl.cos(b)*ufl.cos(g)                       , ufl.cos(b)*ufl.cos(g)                       ]])

def scalar_backmap(flag: bool, u: dolfin.Function, J):
    """
        maps scalar quantity from actual to reference configuration, if flag is True, else returns same quantity

        *Arguments*
            flag: boolean
            u: scalar field (ufl.Function)
            J: jacobian (J = det(F))

        *Example*
            u = volume_mapping(True, u, J)
        """
    return u*J if flag is True else u

def vector_backmap(flag: bool, u, F):
    """
        maps vector quantity from actual to reference configuration, if flag is True, else returns same quantity

        *Arguments*
            flag: boolean
            u: vector field (ufl.VectorFunction)
            F: deformationgradient

        *Example*
            u = vector_mapping(True, u, F)
        """
    return ufl.dot(u, ufl.inv(F.T)) * ufl.det(F) if flag is True else u

def tensor_backmap(flag: bool, flag_complete: bool, U, F):
    """
        maps tensorial quantity from actual to reference configuration, if flag is True, else returns same quantity
        if incomplete first F is not applied

        *Arguments*
            flag: boolean
            flag_incomplete: boolean
            u: tensor field (ufl.TensorFunction)
            F: deformationgradient

        *Example*
            U = tensor_mapping(True, U, F)
        """
    if flag_complete is True:
        P = ufl.inv(F) * U * ufl.inv(F.T) * ufl.det(F) * ufl.det(F)
    else:
        P = U * ufl.inv(F.T) * ufl.det(F) * ufl.det(F)
    return P if flag is True else U

def norm(field: dolfin.Function):
    """
    t.b.d.
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
