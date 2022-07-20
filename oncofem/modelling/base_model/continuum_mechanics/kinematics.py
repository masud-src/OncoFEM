"""
# **************************************************************************#
#                                                                           #
# === Kinematics ===========================================================#
#                                                                           #
# **************************************************************************#
# Definition of kinematic quantities for finite-element calculations
#
# Co-author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
# Co-author: Maximilian Brodbeck <maximilian.brodbeck@isd.uni-stuttgart.de>
#
# --------------------------------------------------------------------------#
"""

import ufl

def calc_defGrad(u):
    """
    calculates the deformation Gradient F = I + grad(u)

    *Arguments*
        u: displacement vector

    *Output*
        gives tensor as ufl class

    *Example*
        F = calc_defGrad(u) 
    """
    return ufl.Identity(len(u)) + ufl.grad(u)

def calc_detDefGrad(u):
    """
    calculates the determinant of deformation Gradient J = det(F)

    *Arguments*
       u: displacement vector

    *Output*
       gives scalar as ufl class

    *Example*
       J = calc_detDefGrad(u) 
    """
    return ufl.det(calc_defGrad(u))

def calc_detDefGrad_lin(u):
    """
    calculates the linearized determinant of deformation gradient
    lin J = 1.0 + tr(lin E)

    *Arguments*
       u: displacement vector

    *Output*
        gives scalar as ufl class

    *Example*
        J = calc_detDefGrad_lin(u) 
    """
    return 1.0 + ufl.tr(calcStrain_GreenLagrange_lin(u))

def calcStrain_CauchyGreenL(u):
    """
    calculates the left Cauchy-Green strain
    B = F*F.T

    *Arguments*
        u: displacement vector

    *Output*
        gives tensor as ufl class

    *Example*
        B = calcStrain_CauchyGreenL(u) 
    """
    F = calc_defGrad(u)
    return F * F.T

def calcStrain_CauchyGreenL_lin():
    """
    T.B.D.
    """
    exit("Not implemented yet")

def calcStrain_CauchyGreenR(u):
    """
    calculates the right Cauchy-Green strain
    C = F.T*F

    *Arguments*
        u: displacement vector

    *Output*
        gives tensor as ufl class

    *Example*
        C = calcStrain_CauchyGreenR(u) 
    """
    F = calc_defGrad(u)
    return F.T * F

def calcStrain_CauchyGreenR_lin():
    """
    T.B.D.
    """
    exit("Not implemented yet")

def calcStrain_GreenLagrange(u):
    """
    calculates the Green Lagrange strain
    E = 0.5*(C-I)

    *Arguments*
        u: displacement vector

    *Output*
        gives tensor as ufl class

    *Example*
        E = calcStrain_GreenLagrange(u) 
    """
    C = calcStrain_CauchyGreenR(u)
    return 0.5 * (C - ufl.Identity(len(u)))

def calcStrain_GreenLagrange_lin(u):
    """
    calculates the linearized Green Lagrange strain
    lin E = 0.5*(nabla u.T + nabla u)

    *Arguments*
        u: displacement vector

    *Output*
        gives tensor as ufl class

    *Example*
        lin_E = calcStrain_GreenLagrange_Lin(u) 
    """
    return ufl.sym(ufl.grad(u))

def calcStrain_Biot(u):
    """
    calculates the Biot strain
    E(1/2) = (sqrt(C)-I)

    *Arguments*
        u: displacement vector

    *Output*
        gives tensor as ufl class

    *Example*
        E_Biot = calcStrain_Biot(u) 
    """
    C = calcStrain_CauchyGreenL(u)
    return ufl.sqrt(C) - ufl.Identity(u)

def calcStrain_Hencky(u):
    """
    calculates the Hencky strain
    E(0) = 0.5*(ln(C))

    *Arguments*
        u: displacement vector

    *Output*
        gives tensor as ufl class

    *Example*
        E_Hencky = calcStrain_Hencky(u) 
    """
    C = calcStrain_CauchyGreenL(u)
    return 0.5 * ufl.ln(C)

def calcStrain_KarniRainer(u):
    """
    calculates the Karni-Rainer tensor
    K = 0.5*(nabla u.T + nabla u)

    *Arguments*
        u: displacement vector

    *Output*
        gives tensor as ufl class

    *Example*
        K = calcStrain_KarniRainer(u) 
    """
    B = calcStrain_CauchyGreenL(u)
    return 0.5*(B - ufl.Identity(len(u)))
