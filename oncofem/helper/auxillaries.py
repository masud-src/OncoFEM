"""
Definition of auxillary helper functions

Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
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
