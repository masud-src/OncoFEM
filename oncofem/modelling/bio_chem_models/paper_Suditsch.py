#!/usr/bin/env python
"""
# **************************************************************************#
#                                                                           #
# === Bio-chemical model  ==================================================#
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

class paper_Suditsch(BioChemModel):
    """asdf"""
    def __init__(self, ip):
        """asdf"""
        super().__init__(ip)
        self.flag_proliferation = False
        self.flag_metabolism = False
        self.flag_agent = False
        self.flag_necrosis = False
        self.flag_angiogenesis = False

        # healthy brain
        self.rhoShR = ip.param.mat.rhoShR
        self.v_Sh_necrosis = 1E-5 * 86400
        # solid tumor
        self.rhoStR = ip.param.mat.rhoStR
        self.nSt_max = 0.4
        self.nSt_init = 8E-7
        self.v_St_necrosis = 1E-5 * 86400
        self.kappa_St_proliferation = 0.35856  # 0.35856
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
        self.Ft_Fn_fac = 1.0e-4
        self.v_Fn_regrowth = 0.0864 * 4
        # agent
        self.molFa = ip.param.add.molFdelta[1]
        self.cFa_min_impact = 5E-9
        self.v_Fa_max = 7.88E-6 * 86400
        self.reaction_rate_cFa = 3484.51
        self.v_Fa = 2.29E-7 * 1.8
        self.v_Fa_halflife = 0.1

    def return_prod_terms(self):
        u, p, nSh, nSt, nSn, cFt, cFn, cFa = self.prim_vars

        H1 = df.conditional(df.gt(cFn, self.cFn_min_growth), 1.0, 0.0)                  # Enough nutrients
        H2 = df.conditional(df.le(cFn, self.cFn_min_necrosis), 1.0, 0.0)                # Start of Necrosis
        H4 = df.conditional(df.ge(cFa, self.cFa_min_impact), 1.0, 0.0)                  # Agent minimal impact
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
        hat_Sh_Fn_loss = df.Constant(0.0)
        hat_Sn_Fn_gain = df.Constant(0.0)
        hat_nSh = df.Constant(0.0)
        hat_nSt = df.Constant(0.0)
        hat_nSn = df.Constant(0.0)
        hat_cFt = df.Constant(0.0)
        hat_cFn = df.Constant(0.0)
        hat_cFa = df.Constant(0.0)

        if self.flag_proliferation:
            hat_St_Fn_gain = H1 * abs(nSt) * self.rhoStR * self.kappa_St_proliferation * (1.0 - (nSt / self.nSt_max))
            hat_Ft_Fn_gain = H1 * nF * cFt * self.molFt * self.kappa_Ft_proliferation * (1.0 - (cFt / self.cFt_threshold))

        if self.flag_metabolism:
            hat_Fn_basal_loss = self.v_In_basal * self.molFn * self.NFt * (nF * cFt * self.molFt + nSt * self.rhoStR)
            hat_Fn_proli_loss = self.f_proli * (hat_Ft_Fn_gain * df.Constant(self.Ft_Fn_fac) + hat_St_Fn_gain)

        if self.flag_necrosis:
            hat_Sh_Fn_loss = H2 * abs(nSh) * self.rhoShR * self.v_Sh_necrosis
            hat_Sn_Fn_gain = hat_Sh_Fn_loss

        if self.flag_agent:
            pass        
        #    hat_St_Fn_gain *= (1.0 - H4)
        #    hat_St_Fa_loss = H4 * abs(nSt) * self.rhoStR * (self.v_Fa_max + cFa * self.reaction_rate_cFa)
        #    hat_Ft_Fa_loss = nF * cFt * self.molFt * H4 * (self.v_Fa_max + cFa * self.reaction_rate_cFa)
        #    hat_Fa_St_loss = self.v_Fa * (hat_St_Fa_loss + hat_Ft_Fa_loss) * H4 + self.v_Fa_halflife * cFa * self.molFa * H9
        #    #hat_Ft_Fn_loss = nF * cFt * self.molFt * H2 * self.v_St_necrosis

        hat_nSh = - hat_Sh_Fn_loss 
        hat_nSt = (abs(hat_St_Fn_gain) - hat_St_Fn_loss - hat_St_Fa_loss) / self.rhoStR
        #hat_nSn = hat_Sn_Fn_gain
        hat_cFt = hat_Ft_Fn_gain - hat_Ft_Fn_loss - hat_Ft_Fa_loss
        hat_cFn = - (hat_Fn_proli_loss + hat_Fn_basal_loss) + hat_Fn_angio_gain + hat_Fn_regain
        hat_cFa = - hat_Fa_St_loss

        prod_list = [None] * (len(self.prim_vars) - 2)
        prod_list[0] = -df.conditional(df.le(nF, 0.3), 1.0, 0.0) * df.Constant(0.1) * nSt / 2.0
        prod_list[1] = hat_nSt - df.conditional(df.le(nF, 0.3), 1.0, 0.0) * df.Constant(0.1) * nSt / 2.0
        prod_list[2] = df.conditional(df.le(nF, 0.3), 1.0, 0.0) * df.Constant(0.1) * nSt
        prod_list[3] = hat_cFt
        prod_list[4] = hat_cFn
        prod_list[5] = hat_cFa
        return prod_list
