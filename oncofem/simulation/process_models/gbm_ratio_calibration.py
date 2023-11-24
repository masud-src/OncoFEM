"""
Definition of bio-chemical model set-up, that is used for simulation in model paper.
"""

import dolfin as df
from .process_model import ProcessModel


class GBMRatioCalibration(ProcessModel):
    def __init__(self):
        super().__init__()
        self.flag_proliferation = False
        self.flag_metabolism = False
        self.flag_necrosis = False
        self.prim_vars = None
        self.dt = None
        # healthy brain
        self.nu_Sh_necrosis = 8.64 * 1e-05 
        self.nSn_max = 0.75
        # solid tumor
        self.nSt_ms = 8.0e-7
        self.nSt_max = 0.75
        self.nSt_init = 1.0e-2
        self.nu_St_proliferation = 4.5856e-7  # 0.35856
        self.nu_St_necrosis = 1E-15 * 86400
        self.nSt_thres_lin_ms = 5e-5
        self.fac_nSt_lin_ms = 1e-1
        # mobile cancer cells
        self.cFt_ms = 7.3e-1
        self.NFt = 1E11
        self.nu_Ft_proliferation = 8.64e7
        self.nu_Ft_necrosis = 0.0 * 86400
        self.cFt_max = 9.828212E-1                  # 10e12 * mol / m^3 
        # nutrients
        self.cFn_min_growth = 0.35
        self.cFn_min_necrosis = 0.95
        self.Kgr = 0.156
        self.nu_Fn_basal = 8.64e-12
        self.nu_Fn_proli = 4.64e-8

    def set_input(self, model):
        self.prim_vars = df.split(model.ansatz_functions)
        self.rhoShR = model.rhoSdeltaR[0]
        self.rhoStR = model.rhoSdeltaR[1]
        self.rhoSnR = model.rhoSdeltaR[2]
        self.molFt = model.molFkappa[0]
        self.molFn = model.molFkappa[1]
        self.dt = model.dt

    def get_output(self):
        u, p, nSh, nSt, nSn, cFt, cFn = self.prim_vars
        nF = 1.0 - (nSh + nSt + nSn)

        hat_Sh_Fn_loss = df.Constant(0.0)  # Necrosis of healthy cells
        hat_St_Fn_gain = df.Constant(0.0)  # Proliferation of solid tumor cells
        hat_St_Fn_loss = df.Constant(0.0)  # Necrosis of solid tumor cells
        hat_Sn_St_gain = df.Constant(0.0)  # Necrosis of solid tumor cells
        hat_Ft_Fn_gain = df.Constant(0.0)  # Proliferation of mobile cancer cells
        hat_Ft_Fn_loss = df.Constant(0.0)  # Necrosis of mobile cancer cells
        hat_Fn_Ft_loss = df.Constant(0.0)  # Metabolism of mobile tumor cells 
        hat_Fn_St_loss = df.Constant(0.0)  # Metabolism of solid tumor cells 

        #if self.flag_metabolism:
        #    hat_Fn_bas_loss = self.nu_In_basal * self.molFn * self.NFt * (nF * cFt * self.molFt + nSt * self.rhoStR)
        #    hat_Fn_pro_loss = self.f_proli * ((self.rhoStR / self.molFt) * hat_Ft_Fn_gain + hat_St_Fn_gain)
        #    hat_cFn = - (hat_Fn_pro_loss + hat_Fn_bas_loss)

        #if self.flag_necrosis:
        #    H2 = df.conditional(df.le(cFn, self.cFn_min_necrosis), 1.0, 0.0)  # Start of Necrosis
        #    # hat_Sh_Fn_loss = H2 * nSh * self.rhoShR * self.nu_Sh_necrosis
        #    hat_St_Fn_loss = H2 * nSt * self.rhoStR * self.nu_St_necrosis
        #    # hat_Ft_Fn_loss = H2 * nF * cFt * self.molFt * self.nu_Ft_necrosis
        #    hat_Sn_Fn_gain = hat_Sh_Fn_loss + hat_St_Fn_loss + hat_Ft_Fn_loss

        if self.flag_proliferation:
            cond_1 = df.gt(cFn, self.cFn_min_growth)    # Enough nutrients
            H3 = df.conditional(df.gt(cFt, self.cFt_ms), 1.0, 0.0)  # mobile cancer cells above threshold

            cFt_enough_nutrients = cFt * df.Constant(self.nu_Ft_proliferation) * (1.0 - cFt / df.Constant(self.cFt_max))
            hat_Ft_Fn_gain = df.conditional(cond_1, cFt_enough_nutrients, df.Constant(0.0))

            nSt_enough_nutrients = H3 * df.Constant(self.nu_St_proliferation) * (1.0 - nSt / (self.nSt_max * 1.1))
            hat_St_Fn_gain = df.conditional(cond_1, nSt_enough_nutrients, df.Constant(0.0))

        if self.flag_metabolism:
            hat_Fn_Ft_loss = cFt * df.Constant(self.nu_Fn_basal)
            hat_Fn_St_loss = nSt * df.Constant(self.nu_Fn_proli)

        if self.flag_necrosis:
            cond_2 = df.lt(cFn, 1.0 - 1.0e-5)  # Enough nutrients
            H2 = df.conditional(cond_2, 1.0, 0.0)  # Enough nutrients
            hat_Sh_Fn_loss = H2 * df.Constant(self.nu_Sh_necrosis) * (1.0 - cFn / (1.0 - 1.0e-5))

        hat_nSh = - hat_Sh_Fn_loss  
        hat_nSt = hat_St_Fn_gain - hat_St_Fn_loss  
        hat_nSn = hat_Sn_St_gain + 0 * hat_Sh_Fn_loss
        hat_cFt = hat_Ft_Fn_gain - hat_Ft_Fn_loss  
        hat_cFn = - hat_Fn_Ft_loss - hat_Fn_St_loss  

        prod_list = [None] * (len(self.prim_vars) - 2)
        prod_list[0] = hat_nSh  # -df.conditional(df.le(nF, 0.3), 1.0, 0.0) * df.Constant(0.1) * nSt / 2.0
        prod_list[1] = hat_nSt  # - df.conditional(df.le(nF, 0.3), 1.0, 0.0) * df.Constant(0.1) * nSt / 2.0
        prod_list[2] = hat_nSn  # df.conditional(df.le(nF, 0.3), 1.0, 0.0) * df.Constant(0.1) * nSt
        prod_list[3] = hat_cFt  # hat_cFt
        prod_list[4] = hat_cFn  # hat_cFn
        return prod_list