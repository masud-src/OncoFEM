"""
Definition of bio-chemical model set-up, that is used for simulation in model paper.
"""
import dolfin as df
import ufl
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
        self.nu_Sh_necrosis = 4.64 * 1e-6
        self.nSn_max = 0.75
        # solid tumor
        self.nSt_ms = 8.0e-7
        self.nSt_max = 0.75
        self.nSt_init = 1.0e-2
        self.nu_St_init = 4.5856e-7
        self.nu_St_proliferation = 8.5856e-6  # 0.35856
        self.nu_St_necrosis = 1E-15 * 86400
        self.nSt_thres_lin_ms = 5e-5
        self.fac_nSt_lin_ms = 1e-1
        # mobile cancer cells
        self.cFt_ms = 7.3e-1
        self.NFt = 1E11
        self.nu_Ft_proliferation = 1.64e8
        self.nu_Ft_necrosis = 0.0 * 86400
        self.cFt_max = 9.828212E-1                  # 10e12 * mol / m^3 
        # nutrients
        self.cFn_min_growth = 0.35
        self.cFn_min_necrosis = 0.95
        self.Kgr = 0.156
        self.nu_Fn_basal = 8.64e-10
        self.nu_Fn_proli = 4.64e-7

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
        hat_Ft_Fn_gain = df.Constant(0.0)
        hat_St_Fn_gain = df.Constant(0.0)
        hat_St_Fn_loss = df.Constant(0.0)  # Necrosis of solid tumor cells
        hat_Sn_St_gain = df.Constant(0.0)  # Necrosis of solid tumor cells
        hat_Ft_Fn_loss = df.Constant(0.0)  # Necrosis of mobile cancer cells
        hat_Fn_Ft_loss = df.Constant(0.0)  # Metabolism of mobile tumor cells
        hat_Fn_St_loss = df.Constant(0.0)  # Metabolism of solid tumor cells

        if self.flag_proliferation:
            cond_1 = ufl.gt(cFn, self.cFn_min_growth)    # Enough nutrients
            cond_2 = ufl.gt(cFt, self.cFt_ms)            # mobile cancer cells above threshold
            cond_3 = ufl.gt(cFt, self.cFt_max)           # mobile cancer cells above maximum
            cond_4 = ufl.gt(nSt, 0.0)               # solid cancer cells are present
            H1 = ufl.conditional(cond_1, df.Constant(1.0), df.Constant(0.0))
            H2 = ufl.conditional(cond_2, df.Constant(1.0), df.Constant(0.0))
            H3 = ufl.conditional(cond_3, df.Constant(0.0), df.Constant(1.0))
            H4 = ufl.conditional(cond_4, df.Constant(0.0), df.Constant(1.0))
            # Proliferation of mobile cancer cells
            hat_Ft_Fn_gain = H1 * H4 * cFt * df.Constant(self.nu_Ft_proliferation) * (1.0 - cFt / df.Constant(self.cFt_max))
            # Proliferation of solid tumor cells
            nSt_init_sigmoid = 1.0 / (1.0 + ufl.exp(-(cFt - self.cFt_ms) / (self.cFt_max - self.cFt_ms)))
            nSt_init = H2 * H3 * df.Constant(self.nu_St_init) * nSt_init_sigmoid
            cond_full = self.binary_sigmoid_condition(cFt, self.cFt_ms, self.cFt_max)
            nSt_full = cond_full * self.nu_St_proliferation * nSt * (1.0 - nSt / df.Constant(self.nSt_max))
            hat_St_Fn_gain = H1 * (nSt_init + nSt_full)

        if self.flag_metabolism:
            hat_Fn_Ft_loss = cFt * df.Constant(self.nu_Fn_basal)
            hat_Fn_St_loss = nSt * df.Constant(self.nu_Fn_proli)

        if self.flag_necrosis:
            cond_2 = ufl.lt(cFn, 1.0 - 1.0e-5)  # Enough nutrients
            H2 = ufl.conditional(cond_2, 1.0, 0.0)  # Enough nutrients
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

    def binary_sigmoid_condition(self, x, t_1, t_2, p=0.5, s=0.5):
        """
        Interpolates
        """
        c = 2.0 / (1.0-s) - 1.0
        def f(x,p):
            return x**c / (p**(c-1.0))
        t_3 = p * (t_2 - t_1)
        cond_1 = ufl.conditional(ufl.lt(x, t_1), 0.0, 1.0)
        cond_2 = ufl.conditional(ufl.ge(x, t_1), ufl.conditional(ufl.lt(x, t_3), f(x,p), 1.0), 1.0)
        cond_3 = ufl.conditional(ufl.ge(x, t_3), ufl.conditional(ufl.lt(x, t_2), 1-f(1-x, 1-p), 1.0), 1.0)

        return cond_1 * cond_2 * cond_3
