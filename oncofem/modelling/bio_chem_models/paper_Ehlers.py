#!/usr/bin/env python
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
        self.flag_apoptose = False
        self.flag_necrosis = False
        self.flag_angiogenesis = False
        self.nSt_max = 0.4
        self.nSt_init = 8E-7
        self.tumor_detection_value = 9E-5
        self.kappa_Ft_proliferation = 0.0864
        self.cFn_min_growth = 0.35
        self.cFt_threshold = 9.828212E-1
        self.cFN_min_necrosis = 0.9 * self.cFn_min_growth
        self.cFd_min_impact = 5E-9
        self.cFv_max = 2.5E-9
        self.cFn_min_VEGF_prod = 0.36
        self.cFv_angio = 2.5E-11
        self.cFv_init = 1E-13
        self.v_In_basal = 8.64E-37  # 17
        self.v_St_growth = 0.35856e-3  # 10
        self.f_proli = 0.864
        self.NFt = 1E11
        self.rhoStR = ip.param.mat.rhoStR
        self.molFt = ip.param.mat.molFt
        self.molFn = ip.param.add.molFdelta[0]
        self.molFv = ip.param.add.molFdelta[1]
        self.molFa = ip.param.add.molFdelta[2]

    def return_prod_terms(self):
        u, p, nSh, nSt, nSn, cFt, cFn, cFv, cFa = self.prim_vars

        H1 = df.conditional(df.gt(cFn, self.cFn_min_growth), 1.0, 0.0)
        H2 = df.conditional(df.le(cFn, self.cFN_min_necrosis), 1.0, 0.0)
        H3 = df.conditional(df.gt(cFt, self.cFt_ms), 1.0, 0.0)
        H4 = df.conditional(df.ge(cFa, self.cFd_min_impact), 1.0, 0.0)
        H5 = df.conditional(df.ge(cFa, 0.9875 * self.cFv_max), 0.0, 1.0)
        H6 = df.conditional(df.gt(cFn, self.cFn_min_VEGF_prod), 0.0, 1.0)
        H7 = df.conditional(df.gt(cFv, self.cFv_angio), 1.0, 0.0)
        H8 = df.conditional(df.ge(cFv, self.cFv_init), 1.0, 0.0)
        H9 = df.conditional(df.gt(cFa, 0.0), 1.0, 0.0)

        nF = 1.0 - (nSh + nSt + nSn)

        hat_St_Fn_gain = H1 * abs(nSt) * self.rhoStR * self.v_St_growth * (1.0 - (nSt / self.nSt_max))
        hat_Ft_Fn_gain = nF * cFt * self.molFt * H1 * self.kappa_Ft_proliferation * (1.0 - (cFt / self.cFt_threshold)) * 1.0
        hat_Fn_basal_loss = self.v_In_basal * self.molFn * self.NFt * (nF * cFt * self.molFt + nSt * self.rhoStR)
        hat_Fn_proli_loss = self.f_proli * hat_Ft_Fn_gain * 1E-14

        hat_St = hat_St_Fn_gain
        hat_Ft = hat_Ft_Fn_gain
        hat_Fn = - (hat_Fn_basal_loss + hat_Fn_proli_loss)

        prod_list = [None] * (len(self.prim_vars) - 2)
        prod_list[1] = hat_St
        prod_list[3] = hat_Ft
        # prod_list[4] = hat_Fn
        return prod_list
