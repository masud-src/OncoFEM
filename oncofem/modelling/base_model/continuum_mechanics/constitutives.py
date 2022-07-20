"""
# *************************************************************************#
#                                                                          #
# === Constitutives =======================================================#
#                                                                          #
# *************************************************************************#
# Definition of constitutive quantities for finite-element calculations
#
# Co-author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
# Co-author: Maximilian Brodbeck <maximilian.brodbeck@isd.uni-stuttgart.de>
#
# --------------------------------------------------------------------------#
"""

import dolfin
import optifen.kinematics as kin
import ufl

# **************************************************************************#
#      Classes                                                              #
# **************************************************************************#
# Definition of Classes

# --------------------------------------------------------------------------#
class Constitutives:
    pass

# **************************************************************************#
#      Functions                                                            #
# **************************************************************************#
# Definition of Functions

# --------------------------------------------------------------------------#
def calc_DarcyVelocity(k_f, p):
        """
        calculates darcy velocity

        *Arguments:*
                k_f: Permeability
                p: pressure scalarr

        *Example:*
                calc_DarcyVelocity(1.0e-6, p)
        """
        return - k_f * ufl.grad(p)

def calc_DarcyVelocity_MassExchange(kF, gammaFR, nF, hatnF, rhoFR, p, v_S):
        """
        calculates darcy velocity

        *Arguments:*
                k_f: Permeability
                p: pressure scalarr

        *Example:*
                calc_DarcyVelocity(1.0e-6, p)
        """
        nFw_FS = dolfin.conditional(ufl.sqrt(hatnF*hatnF)>dolfin.DOLFIN_EPS, (kF / gammaFR + (nF*nF) / (hatnF * rhoFR) ) * (- ufl.grad(p) - hatnF / nF * rhoFR * v_S), (kF / gammaFR) * (- ufl.grad(p)))
        return nFw_FS

def calcStress_NeoHookean_Kirchhoff(u, lambda_, mu_):
        """
        calaulates Stress with neohookean material law  and Kirchhoff stress

        *Arguments:*
                u: displacement vector (2D/3D)
                lambda_: first lamé parameter
                mu_: second lamé parameter

        *Example:*
                calcStress_NeoHookean_Kirchhoff(u, 1e7, 1e7)
        """
        return 2.0 * mu_ * ufl.calcStrain_GreenLagrange(u) + lambda_ * ufl.tr(kin.calcStrain_GreenLagrange(u)) * ufl.Identity(len(u))

def calcStress_NeoHookean_PiolaKirchhoff2_lin(u, lambda_, mu_):
        """
        calculates linearized Stress with neohookean material law and Piola-Kirchhoff 2 stress

        *Arguments:*
                u: displacement vector (2D/3D)
                lambda_: first lamé parameter
                u_: second lamé parameter

        *Example:*
                calcStress_NeoHookean_PiolaKirchhoff2_lin(u, 1e7, 1e7)
        """
        return 2.0 * mu_ * kin.calcStrain_GreenLagrange_lin(u) + lambda_ * ufl.tr(kin.calcStrain_GreenLagrange_lin(u)) * ufl.Identity(len(u))

def calcStressExtra_NeoHookean_PiolaKirchhoff2_lin(u, p, lambda_, mu_):
        """
        calculates linearized extra Stress with neohookean material law and Piola-Kirchhoff 2 stress

        *Arguments:*
                u: displacement vector (2D/3D)
                p: pressure scalar
                lambda_: first lamé parameter
                u_: second lamé parameter

        *Example:*
                calcStressExtra_NeoHookean_PiolaKirchhoff2_lin(u, p, 1e7, 1e7)
        """
        T = calcStress_NeoHookean_PiolaKirchhoff2_lin(u,lambda_,mu_)
        lin_J = kin.calc_detDefGrad_lin(u)
        return T - p * (lin_J * ufl.Identity(len(u))) - 2.0 * kin.calcStrain_GreenLagrange_lin(u) 

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
        tau_xy = T[1,0] * T[1,0] 
        if ufl.shape(T)[0]==2:
                return ufl.sqrt(sig2_x + sig2_y - sig_x * sig_y + 3.0 * tau_xy)
        elif ufl.shape(T)[0]==3:
                sig_z = T[2,2]
                sig2_z = sig_z * sig_z
                tau_xz = T[0,2] * T[0,2]
                tau_yz = T[1,2] * T[1,2]
                return ufl.sqrt(sig2_x + sig2_y + sig2_z - sig_x * sig_y - sig_x * sig_z - sig_y * sig_z + 3.0 * (tau_xy + tau_xz + tau_yz))

def calcGrowth_modVerhulst(u, real_dens_u, kappa, u_max, dt):
        """
        Calculates a modified Verhulst kinetic

        *Parameters:*
                u: field that grows
                real_dens_u: real density of field u
                kappa: growth parameter
                u_max: maximal Value of field u
                dt: time step

        *Example:*
                calcGrowth_modVerhulst(u, 1000.0, 0.5. 2000.0, 1.0)
        """
        return u * real_dens_u * kappa * (1.0 - u / u_max) * dt

def calcGrowth_modMonod(u, mu, K, cIn, cIn_min):
        """
        Calculates a modified Monod kinetic

        *Parameters:*
                u: field that grows
                mu: growth parameter
                K: growth parameter
                cIn: depending field
                cIn_min: minimal value of depending field
        *Example:*
                calcGrowth_modMonod(u, 0.5, 1.0, cIn, 0.78)
        """
        return u * mu * (cIn - cIn_min) / (K + cIn - cIn_min)
