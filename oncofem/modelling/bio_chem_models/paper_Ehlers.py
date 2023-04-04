"""
# **************************************************************************#
#                                                                           #
# === Bio-chemical models ==================================================#
#                                                                           #
# **************************************************************************#
# Definition of bio-chemical model class
#
# Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
#
# --------------------------------------------------------------------------#
"""

import dolfin as df
from .bio_chem_models import BioChemModel

class paper_Ehlers(BioChemModel):
    """asdf"""
    def __init__(self, ip):
        """asdf"""
        super().__init__(ip)
        self.flag_proliferation = False
        self.flag_metabolism = False
        self.flag_agent = False
        self.flag_necrosis = False
        self.flag_angiogenesis = False

        self.tumor_detection_value = 9E-5
        # solid tumor
        self.rhoStR = ip.param.mat.rhoStR
        self.nSt_max = 0.4
        self.nSt_init = 8E-7
        self.v_St_necrosis = 1E-5 * 86400
        # mobile cancer cells
        self.molFt = ip.param.mat.molFt
        self.kappa_Ft_proliferation = 0.0864
        self.cFt_threshold = 9.828212E-13
        self.NFt = 1E11
        # nutrients
        self.molFn = ip.param.add.molFdelta[0]
        self.cFn_min_growth = 0.35
        self.cFn_min_VEGF_prod = 0.36
        self.cFn_min_necrosis = 0.9 * self.cFn_min_growth
        self.cFn_angio = 0.55
        self.Kgr = 0.156
        self.v_In_basal = 8.64E-17
        self.f_proli = 0.864
        self.v_Fn_regrowth = 0.0864 * 4
        self.v_St_growth = 0.35856
        # VEGF
        self.molFv = ip.param.add.molFdelta[1]
        self.cFv_init = 1E-13
        self.cFv_angio = 2.5E-11
        self.cFv_max = 2.5E-9
        self.v_Fv_plus = 0.155E-11 * 86400
        # agent
        self.molFa = ip.param.add.molFdelta[2]
        self.cFa_min_impact = 5E-9
        self.v_Fa_max = 7.88E-6 * 86400
        self.reaction_rate_cFa = 3484.51
        self.v_Fa = 2.29E-7 * 1.8
        self.v_Fa_halflife = 0.1

    def return_prod_terms(self):
        u, p, nSh, nSt, nSn, cFt, cFn, cFv, cFa = self.prim_vars

        H1 = df.conditional(df.gt(cFn, self.cFn_min_growth), 1.0, 0.0)                  # Enough nutrients
        H2 = df.conditional(df.le(cFn, self.cFn_min_necrosis), 1.0, 0.0)                # Start of Necrosis
        H3 = df.conditional(df.gt(cFt, self.cFt_ms), 1.0, 0.0)                          # Metastatic switch
        H4 = df.conditional(df.ge(cFa, self.cFa_min_impact), 1.0, 0.0)                  # Agent minimal impact
        H5 = df.conditional(df.ge(cFa, 0.9875 * self.cFv_max), 0.0, 1.0)                # VEGF and Agent
        H6 = df.conditional(df.gt(cFn, self.cFn_min_VEGF_prod), 0.0, 1.0)               # VEGF minimal
        H7 = df.conditional(df.gt(cFv, self.cFv_angio), 1.0, 0.0)                       # Angiogenesis
        H8 = df.conditional(df.ge(cFv, self.cFv_init), 1.0, 0.0)                        # Send out VEGF
        H9 = df.conditional(df.gt(cFa, 0.0), 1.0, 0.0)                                  # Agent

        nF = 1.0 - (nSh + nSt + nSn)

        hat_Ft_Fn_gain = df.Constant(0.0)
        hat_St_Fn_gain = df.Constant(0.0)
        hat_St_Fn_loss = df.Constant(0.0)
        hat_St_Fa_loss = df.Constant(0.0)
        hat_Ft_Fn_loss = df.Constant(0.0)
        hat_Ft_Fa_loss = df.Constant(0.0)
        hat_Fn_proli_loss = df.Constant(0.0)
        hat_Fn_basal_loss = df.Constant(0.0)
        hat_Fn_angio_gain = df.Constant(0.0)
        hat_Fn_regain = df.Constant(0.0)
        hat_Fv_St_gain = df.Constant(0.0)
        hat_Fa_St_loss = df.Constant(0.0)

        if self.flag_proliferation:
            hat_St_Fn_gain = H1 * abs(nSt) * self.rhoStR * self.v_St_growth
            hat_Ft_Fn_gain = H1 * nF * cFt * self.molFt * self.kappa_Ft_proliferation * (1.0 - (cFt / self.cFt_threshold))

        if self.flag_agent:
            hat_St_Fn_gain *= (1.0 - H4)
            hat_St_Fa_loss = H4 * abs(nSt) * self.rhoStR * (self.v_Fa_max + cFa * self.reaction_rate_cFa)
            hat_Ft_Fa_loss = nF * cFt * self.molFt * H4 * (self.v_Fa_max + cFa * self.reaction_rate_cFa)
            hat_Fa_St_loss = self.v_Fa * (hat_St_Fa_loss + hat_Ft_Fa_loss) * H4 + self.v_Fa_halflife * cFa * self.molFa * H9

        if self.flag_necrosis:
            hat_St_Fn_loss = H2 * abs(nSt) * self.rhoStR * self.v_St_necrosis
            hat_Ft_Fn_loss = nF * cFt * self.molFt * H2 * self.v_St_necrosis

        if self.flag_angiogenesis:
            hat_St_Fn_gain *= (H7 + (1 - H7) * ((cFn - self.cFn_min_growth) / (self.Kgr + cFn - self.cFn_min_growth)))
            hat_Fn_angio_gain = self.f_proli * hat_St_Fn_gain  # * angio_activate
            hat_Fn_regain = nF * self.molFn * cFn * self.v_Fn_regrowth * (1 - cFn) * (H4 + 0.1 * H7)
            hat_Fv_St_gain = nF * abs(nSt) * self.molFv * self.v_Fv_plus * H8 * ((1.0 - cFn) / (1.0 - cFn + self.cFn_min_growth)) * (1 - (cFv / self.cFv_max)) * H5

        if self.flag_metabolism:
            hat_Fn_basal_loss = self.v_In_basal * self.molFn * self.NFt * (nF * cFt * self.molFt + nSt * self.rhoStR)
            hat_Fn_proli_loss = self.f_proli * (hat_Ft_Fn_gain * 1E-4 + hat_St_Fn_gain)

        hat_St = (abs(hat_St_Fn_gain) - hat_St_Fn_loss - hat_St_Fa_loss) / self.rhoStR
        hat_Ft = hat_Ft_Fn_gain - hat_Ft_Fn_loss - hat_Ft_Fa_loss
        hat_Fn = - (hat_Fn_proli_loss + hat_Fn_basal_loss) + hat_Fn_angio_gain + hat_Fn_regain
        hat_Fv = hat_Fv_St_gain
        hat_Fa = - hat_Fa_St_loss

        prod_list = [None] * (len(self.prim_vars) - 2)
        prod_list[1] = hat_St
        prod_list[3] = hat_Ft
        prod_list[4] = hat_Fn
        prod_list[5] = hat_Fv
        prod_list[6] = hat_Fa
        return prod_list
