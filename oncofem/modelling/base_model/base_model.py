"""
# **************************************************************************#
#                                                                           #
# === Base Model ===========================================================#
#                                                                           #
# **************************************************************************#
# Definition of base model class
#
# Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
#
# --------------------------------------------------------------------------#
"""

import oncofem as of
import ufl

# **************************************************************************#
#      Classes                                                              #
# **************************************************************************#
# Definition of Classes

class BaseModel:
    pass

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
