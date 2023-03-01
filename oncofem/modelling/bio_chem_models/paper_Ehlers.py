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

from .bio_chem_models import BioChemModel

class paper_Ehlers(BioChemModel):
    """asdf"""
    def __init__(self):
        """asdf"""
        super().__init__()
        self.nSt_max = 0.4
        self.nSt_init = 8E-7
        self.tumor_detection_value = 9E-5
        self.kappa_Ft_proliferation = 0.0864
        self.cFn_min_growth = 0.35
        self.cFt_threshold = 9.828212E-1
        self.cFN_min_necrosis = None
        self.cFd_min_impact = None
        self.cFv_max = None
        self.cFn_min_VEGF_prod = None
        self.cFv_angio = None
        self.cFv_init = None
        self.v_In_basal = 8.64E-37  # 17
        self.v_St_growth = 0.35856e-3  # 10
        self.f_proli = 0.864
        self.NFt = 1E11

    def set_prim_vars(self, s):
        pass

    def return_prod_terms(self):
        self.nF = 1.0 - (nSh + nSt + nSn)
        nSt_max = 0.4
        nSt_init = 8E-7
        tumor_detection_value = 9E-5
        kappa_Ft_proliferation = 0.0864
        cFn_min_growth = 0.35
        cFt_threshold = 9.828212E-1
        cFN_min_necrosis = None
        cFd_min_impact = None
        cFv_max = None
        cFn_min_VEGF_prod = None
        cFv_angio = None
        cFv_init = None
        nF = 1.0 - (nSh + nSt + nSn)
        v_In_basal = 8.64E-37  # 17
        v_St_growth = 0.35856e-3  # 10
        f_proli = 0.864
        NFt = 1E11

        H1 = df.conditional(df.gt(cFn, cFn_min_growth), 1.0, 0.0)
        H2 = df.conditional(df.le(cFn, cFN_min_necrosis), 1.0, 0.0)
        H3 = df.conditional(df.gt(cFt, x.param.mat.cFt_ms), 1.0, 0.0)
        H4 = df.conditional(df.ge(cFa, cFd_min_impact), 1.0, 0.0)
        H5 = df.conditional(df.ge(cFa, 0.9875 * cFv_max), 0.0, 1.0)
        H6 = df.conditional(df.gt(cFn, cFn_min_VEGF_prod), 0.0, 1.0)
        H7 = df.conditional(df.gt(cFv, cFv_angio), 1.0, 0.0)
        H8 = df.conditional(df.ge(cFv, cFv_init), 1.0, 0.0)
        H9 = df.conditional(df.gt(cFa, 0.0), 1.0, 0.0)

        hat_St_Fn_gain = H1 * abs(nSt) * x.param.mat.rhoStR * v_St_growth * (1.0 - (nSt / nSt_max))
        hat_Ft_Fn_gain = nF * cFt * x.param.mat.molFt * H1 * kappa_Ft_proliferation * (
                1.0 - (cFt / cFt_threshold)) * 1.0
        hat_Fn_basal_loss = v_In_basal * molFn * NFt * (nF * cFt * x.param.mat.molFt + nSt * x.param.mat.rhoStR)
        hat_Fn_proli_loss = f_proli * hat_Ft_Fn_gain * 1E-14

        hat_St = hat_St_Fn_gain
        hat_Ft = hat_Ft_Fn_gain
        hat_Fn = - (hat_Fn_basal_loss + hat_Fn_proli_loss)

        prod_list = [None] * (len(model.prim_vars_list) - 2)
        prod_list[1] = hat_St
        prod_list[3] = hat_Ft
        # prod_list[4] = hat_Fn
        return prod_list
