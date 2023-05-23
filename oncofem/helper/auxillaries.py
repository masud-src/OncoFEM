"""
Definition of auxillary helper functions

Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""

import dolfin
import ufl
import pprint
import numpy as np
from numpy import logical_and as l_and, logical_not as l_not
from scipy.spatial.distance import directed_hausdorff
from oncofem.helper import constant as const

class BoundingBox(dolfin.SubDomain):
    """

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
    return MapAverageMaterialProperty(params, distributions, weights)

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

def meshfunction_2_function(mf: dolfin.MeshFunction, fs: dolfin.FunctionSpace):
    """
    maps meshfunction to functionspace. Only works with constant meshfunction space and linear functionspace
    """
    v2d = dolfin.vertex_to_dof_map(fs)
    u = dolfin.Function(fs)
    u.vector()[v2d] = mf.array()
    return u

def count_parameters(model):
    """
    Count trainable parameters of neural network
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_metrics(preds, targets, patient, tta=False):
    """

    Parameters
    ----------
    preds:
        torch tensor of size 1*C*Z*Y*X
    targets:
        torch tensor of same shape
    patient :
        The patient ID
    tta:
        is tta performed for this run
    """
    pp = pprint.PrettyPrinter(indent=4)
    assert preds.shape == targets.shape, "Preds and targets do not have the same size"

    labels = ["ET", "TC", "WT"]

    metrics_list = []

    for i, label in enumerate(labels):
        metrics = dict(patient_id=patient, label=label, tta=tta, )

        if np.sum(targets[i]) == 0:
            print(f"{label} not present for {patient}")
            sens = np.nan
            dice = 1 if np.sum(preds[i]) == 0 else 0
            tn = np.sum(l_and(l_not(preds[i]), l_not(targets[i])))
            fp = np.sum(l_and(preds[i], l_not(targets[i])))
            spec = tn / (tn + fp)
            haussdorf_dist = np.nan

        else:
            preds_coords = np.argwhere(preds[i])
            targets_coords = np.argwhere(targets[i])
            haussdorf_dist = directed_hausdorff(preds_coords, targets_coords)[0]

            tp = np.sum(l_and(preds[i], targets[i]))
            tn = np.sum(l_and(l_not(preds[i]), l_not(targets[i])))
            fp = np.sum(l_and(preds[i], l_not(targets[i])))
            fn = np.sum(l_and(l_not(preds[i]), targets[i]))

            sens = tp / (tp + fn)
            spec = tn / (tn + fp)

            dice = 2 * tp / (2 * tp + fp + fn)

        metrics[const.HAUSSDORF] = haussdorf_dist
        metrics[const.DICE] = dice
        metrics[const.SENS] = sens
        metrics[const.SPEC] = spec
        pp.pprint(metrics)
        metrics_list.append(metrics)

    return metrics_list