"""
Definition of bio-chemical model set-up, that is used for simulation in
model paper.

Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""

import dolfin as df
from .bio_chem_models import BioChemModel


class ResectionModel(BioChemModel):
    def __init__(self):
        super().__init__()
        self.flag_proliferation = False
        self.flag_metabolism = False
        self.flag_necrosis = False
        self.flag_drug = False

        # healthy brain
        self.nu_Sh_necrosis = 1E-5 * 86400
        # solid tumor
        self.nSt_ms = 8E-7
        self.nSt_max = 0.4
        self.nSt_init = 8E-7
        self.nu_St_proliferation = 0.35856  # 0.35856
        self.nu_St_necrosis = 1E-5 * 86400
        self.nSt_thres_lin_ms = 0.005
        self.fac_nSt_lin_ms = 1e1
        # mobile cancer cells
        self.cFt_ms = 7.3E-1
        self.NFt = 1E11
        self.nu_Ft_proliferation = 0.0864
        self.nu_Ft_necrosis = 1E-5 * 86400
        self.cFt_max = 9.828212E-13
        # nutrients
        self.cFn_min_growth = 0.35
        self.cFn_min_necrosis = 0.5 * self.cFn_min_growth
        self.Kgr = 0.156
        self.nu_In_basal = 8.64E-17
        self.f_proli = 0.864
        # drug
        self.MFd = 93
        self.a_d = 800
        # cFd_infusion_field = df.Expression(("cFd_infusion_concentration*exp(-a*(pow((x[0]-x_source),2)+pow((x[1]-y_source),2)))"), degree=2, cFd_infusion_concentration=cFd_infusion_concentration , a=a_d, x_source=x_source, y_source=y_source)
        self.cFd_min_impact = 5E-9
        self.v_Fd_max = 7.88E-6 * 86400
        self.reaction_rate_cFd = 3484.51
        self.v_Fd = 2.29E-7 * 1.8
        self.v_Fd_halflife = 0.1

    def set_param(self, ip):
        self.rhoShR = ip.param.mat.rhoShR
        self.rhoStR = ip.param.mat.rhoStR
        self.molFt = ip.param.mat.molFt
        self.molFn = ip.param.add.molFkappa[0]
        self.molFd = ip.param.add.molFkappa[1]

    def return_prod_terms(self):
        u, p, nSh, nSt, nSn, cFt, cFn, cFd = self.prim_vars
        nF = 1.0 - (nSh + nSt + nSn)

        hat_Ft_Fn_gain = df.Constant(0.0)
        hat_St_Fn_gain = df.Constant(0.0)
        if self.flag_proliferation:
            H1 = df.conditional(df.gt(cFn, self.cFn_min_growth), 1.0, 0.0)  # Enough nutrients
            H3 = df.conditional(df.gt(cFt, self.cFt_ms), 1.0, 0.0)  # mobile cancer cells above threshold
            # Translation of micrometastatic switch 
            H4 = df.conditional(df.gt(nSt, self.nSt_thres_lin_ms),
                                nSt * self.rhoStR * self.nu_St_proliferation, self.fac_nSt_lin_ms * cFt * self.nu_St_proliferation)

            fac_max_cFt = df.conditional(df.gt(1.0 - (cFt / self.cFt_max), 0.0), 1.0 - (cFt / self.cFt_max), 0.0)
            fac_max_nSt = df.conditional(df.gt(1.0 - (nSt / self.nSt_max), 0.0), 1.0 - (nSt / self.nSt_max), 0.0)
            fac_cFn_min = (cFn - self.cFn_min_growth) / (self.Kgr + (cFn - self.cFn_min_growth))
            hat_Ft_Fn_gain = H1 * nF * cFt * self.molFt * self.nu_Ft_proliferation * fac_cFn_min * fac_max_cFt
            hat_St_Fn_gain = H1 * H3 * H4 * fac_cFn_min * fac_max_nSt

        hat_cFn = df.Constant(0.0)
        if self.flag_metabolism:
            hat_Fn_bas_loss = self.nu_In_basal * self.molFn * self.NFt * (nF * cFt * self.molFt + nSt * self.rhoStR)
            hat_Fn_pro_loss = self.f_proli * ((self.rhoStR / self.molFt) * hat_Ft_Fn_gain + hat_St_Fn_gain)
            hat_cFn = - (hat_Fn_pro_loss + hat_Fn_bas_loss)

        hat_St_Fn_loss = df.Constant(0.0)
        hat_Ft_Fn_loss = df.Constant(0.0)
        hat_Sh_Fn_loss = df.Constant(0.0)
        hat_Sn_Fn_gain = df.Constant(0.0)
        if self.flag_necrosis:
            H2 = df.conditional(df.le(cFn, self.cFn_min_necrosis), 1.0, 0.0)  # Start of Necrosis
            # hat_Sh_Fn_loss = H2 * nSh * self.rhoShR * self.nu_Sh_necrosis
            hat_St_Fn_loss = H2 * nSt * self.rhoStR * self.nu_St_necrosis
            # hat_Ft_Fn_loss = H2 * nF * cFt * self.molFt * self.nu_Ft_necrosis
            hat_Sn_Fn_gain = hat_Sh_Fn_loss + hat_St_Fn_loss + hat_Ft_Fn_loss

        hat_Fd_St_loss = df.Constant(0.0)
        if self.flag_drug:
            H9 = df.conditional(df.gt(cFd, 0.0), 1.0, 0.0)
            hat_St_Fd_loss = H4 * abs(nSt) * self.rhoStR * (self.v_Fd_max + cFd * self.reaction_rate_cFd)
            hat_Ft_Fd_loss = nF * cFt * self.molFt * H4 * (self.v_Fd_max + cFd * self.reaction_rate_cFd)
            hat_Fd_St_loss = self.v_Fd * (hat_St_Fd_loss + hat_Ft_Fd_loss) * H4 + self.v_Fd_halflife * cFd * self.molFd * H9

        hat_nSh = hat_Sh_Fn_loss
        hat_nSt = hat_St_Fn_gain - hat_St_Fn_loss
        hat_nSn = hat_Sn_Fn_gain
        hat_cFt = hat_Ft_Fn_gain - hat_Ft_Fn_loss
        hat_cFd = - hat_Fd_St_loss

        prod_list = [None] * (len(self.prim_vars) - 2)
        prod_list[0] = -df.conditional(df.le(nF, 0.3), 1.0, 0.0) * df.Constant(0.1) * nSt / 2.0
        prod_list[1] = - df.conditional(df.le(nF, 0.3), 1.0, 0.0) * df.Constant(0.1) * nSt / 2.0
        prod_list[2] = df.conditional(df.le(nF, 0.3), 1.0, 0.0) * df.Constant(0.1) * nSt
        prod_list[3] = hat_cFt
        prod_list[4] = hat_cFn
        prod_list[5] = hat_cFd
        return prod_list
