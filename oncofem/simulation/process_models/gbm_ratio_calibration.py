"""
Definition of bio-chemical model set-up, that is used for simulation in model paper.
"""
import dolfin as df
import ufl
from .process_model import ProcessModel


class GBMRatioCalibration(ProcessModel):
    def __init__(self):
        super().__init__()
        self.prim_vars = None
        self.dt = None
        # solid tumor
        self.nSt_ms = 8.0e-7
        self.nSt_max = 0.9
        self.nu_St_proliferation = 8.5856e-7 #1.5856e-5  # 0.35856
        # necrotic tumour
        self.nSn_max = 0.9
        self.nu_Sn_proliferation = 8.5856e-7  # 1.5856e-5  # 0.35856
        # mobile cancer cells
        self.cFt_max = 9.828212E-1                  # 10e12 * mol / m^3 
        self.cFt_ms = self.cFt_max * 0.25
        self.nu_Ft_proliferation = 6.64e7
        # nutrients
        self.cFn_min_growth = 0.35
        self.nu_Fn_basal = 8.64e-10
        self.nu_Fn_proli = 0.64e-6

    def set_input(self, model):
        self.prim_vars = df.split(model.ansatz_functions)
        self.rhoShR = model.rhoSkappaR[0]
        self.rhoStR = model.rhoSkappaR[1]
        self.rhoSnR = model.rhoSkappaR[2]
        self.molFt = model.molFdelta[0]
        self.molFn = model.molFdelta[1]
        self.dt = model.dt

    def get_output(self):
        u, p, nSh, nSt, nSn, cFt, cFn = self.prim_vars
        nS = nSh + nSt + nSn
        nF = 1.0 - nS

        # cFt is larger than threshold and tumour begins to grow
        cond_1 = ufl.gt(cFt, 0.1)                               
        H1 = ufl.conditional(cond_1, df.Constant(1.0), df.Constant(0.0))

        # nS is below max value but will begin with necrosis
        cond_2 = ufl.lt(nSt, 0.9)
        H2 = ufl.conditional(cond_2, df.Constant(1.0), df.Constant(0.0))

        # nS above treshold so healthy cells don't like
        cond_3 = ufl.gt(nS, 0.9)
        H3 = ufl.conditional(cond_3, df.Constant(1.0), df.Constant(0.0))

        # Cells become necrotic
        cond_4 = ufl.lt(cFn, 0.4)
        H4 = ufl.conditional(cond_4, df.Constant(1.0), df.Constant(0.0))

        # Proliferation of mobile cancer cells
        hat_Ft_Fn_gain = cFt * df.Constant(5.8e7) * (1.0 - cFt / df.Constant(self.cFt_max))

        # Proliferation of tumour
        nSt_gain = nSt * df.Constant(3.2e-5)
        nSt_H = ufl.conditional(ufl.gt(nSt, 1e-2), nSt_gain, 1e-6)
        hat_St_Fn_gain = H1 * H2 * (1.0 - H4) * nSt_H * (1.0 - nSt / df.Constant(0.9))

        # Metabolism
        cond_5 = ufl.gt(cFn, 0.0)
        H5 = ufl.conditional(cond_5, df.Constant(1.0), df.Constant(0.0))
        hat_cFn = - H5 * (df.Constant(0.5e0) * nSt * self.rhoStR * 0 + df.Constant(0.5e0) * (nSt + cFt) * self.rhoStR)

        # Necrosis
        nSn_gain = (df.Constant(0.4) - cFn) * df.Constant(1e-5)
        hat_Sn_gain = H4 * nSn_gain * (1.0 - nSn / df.Constant(0.9))       

        hat_nSh = - H3 * hat_St_Fn_gain
        hat_nSt = hat_St_Fn_gain - hat_Sn_gain
        hat_nSn = hat_Sn_gain
        hat_cFt = hat_Ft_Fn_gain

        prod_list = [None] * (len(self.prim_vars) - 2)
        prod_list[0] = hat_nSh              # -df.conditional(df.le(nF, 0.3), 1.0, 0.0) * df.Constant(0.1) * nSt / 2.0
        prod_list[1] = hat_nSt              # - df.conditional(df.le(nF, 0.3), 1.0, 0.0) * df.Constant(0.1) * nSt / 2.0
        prod_list[2] = hat_nSn              # df.conditional(df.le(nF, 0.3), 1.0, 0.0) * df.Constant(0.1) * nSt
        prod_list[3] = hat_cFt              # hat_cFt
        prod_list[4] = hat_cFn              # hat_cFn
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

#hat_nSh = - H2 * (hat_St_Fn_gain + nSh / (nSh+nSt) * hat_Sn_St_gain) 
#        hat_nSt = hat_St_Fn_gain - nSt / (nSh+nSt) * hat_Sn_St_gain
#        hat_nSn = hat_Sn_St_gain
#        hat_cFt = hat_Ft_Fn_gain

# nSt has reached a size where necrosis happens
#        cond_3 = ufl.gt(nSt, 0.60)  # nSt_tres

#        cond_4 = ufl.gt(nSt, 0.005)  # nSt_init
#        cond_5 = ufl.gt(nSn, 0.002)  # nSn_init
#        H3 = ufl.conditional(cond_4, df.Constant(0.0), df.Constant(1.0))  # nSt_tres
#        H4 = ufl.conditional(ufl.Or(cond_3, cond_5), df.Constant(1.0), df.Constant(0.0))
#        hat_Sn_St_gain = H4 * df.Constant(self.nu_Sn_proliferation) * (1.0 - nSn / df.Constant(self.nSn_max))

#    def get_output(self):
#        u, p, nSh, nSt, nSn, cFt, cFn = self.prim_vars
#        nF = 1.0 - (nSh + nSt + nSn)
#
#        hat_Sh_Fn_loss = df.Constant(0.0)  # Necrosis of healthy cells
#        hat_Ft_Fn_gain = df.Constant(0.0)
#        hat_St_Fn_gain = df.Constant(0.0)
#        hat_St_Fn_loss = df.Constant(0.0)  # Necrosis of solid tumor cells
#        hat_Sn_St_gain = df.Constant(0.0)  # Necrosis of solid tumor cells
#        hat_Ft_Fn_loss = df.Constant(0.0)  # Necrosis of mobile cancer cells
#        hat_Fn_Ft_loss = df.Constant(0.0)  # Metabolism of mobile tumor cells
#        hat_Fn_St_loss = df.Constant(0.0)  # Metabolism of solid tumor cells
#
#        if self.flag_proliferation:
#            cond_1 = ufl.gt(cFn, self.cFn_min_growth)   # Enough nutrients
#            cond_2 = ufl.gt(cFt, self.cFt_ms)           # mobile cancer cells above threshold
#            cond_3 = ufl.gt(cFt, self.cFt_max)          # mobile cancer cells above maximum
#            cond_4 = ufl.gt(nSt, 0.1)                   # solid cancer cells are present
#            H1 = ufl.conditional(cond_1, df.Constant(1.0), df.Constant(0.0))
#            H2 = ufl.conditional(cond_2, df.Constant(1.0), df.Constant(0.0))
#            H3 = ufl.conditional(cond_3, df.Constant(0.0), df.Constant(1.0))
#            H4 = ufl.conditional(cond_4, nSt, self.nSt_init)
#            # Proliferation of mobile cancer cells
#            hat_Ft_Fn_gain = H1 * cFt * df.Constant(self.nu_Ft_proliferation) * (1.0 - cFt / df.Constant(self.cFt_max))
#            # Proliferation of solid tumor cells
#            #nSt_init_sigmoid = 1.0 / (1.0 + ufl.exp(-(cFt - self.cFt_ms) / (self.cFt_max - self.cFt_ms)))
#            #nSt_init = H2 * df.Constant(self.nu_St_init) * nSt_init_sigmoid
#            #cond_full = self.binary_sigmoid_condition(cFt, self.cFt_ms, self.cFt_max)
#            #nSt_full = cond_full * self.nu_St_proliferation * nSt * (1.0 - nSt / df.Constant(self.nSt_max))
#            hat_St_Fn_gain = H1 * H2 * df.Constant(self.nu_St_proliferation) * (1.0 - nSt / df.Constant(self.nSt_max))
#            hat_Sn_St_gain = H4 * df.Constant(self.nu_Sh_necrosis) * (1.0 - cFn / (1.0 - 1.0e-5))
#
#        #if self.flag_metabolism:
#        #    hat_Fn_Ft_loss = cFt * df.Constant(self.nu_Fn_basal)
#        #    hat_Fn_St_loss = nSt * df.Constant(self.nu_Fn_proli)
#
#        #if self.flag_necrosis:
#        #    cond_2 = ufl.lt(cFn, 1.0 - 1.0e-5)  # Enough nutrients
#        #    H2 = ufl.conditional(cond_2, 1.0, 0.0)  # Enough nutrients
#        #    hat_Sh_Fn_loss = H2 * df.Constant(self.nu_Sh_necrosis) * (1.0 - cFn / (1.0 - 1.0e-5))
#
#        hat_nSh = - hat_Sn_St_gain - 0.8 * hat_St_Fn_gain
#        hat_nSt = hat_St_Fn_gain #- hat_St_Fn_loss  
#        hat_nSn = hat_Sn_St_gain #+ hat_Sh_Fn_loss
#        hat_cFt = hat_Ft_Fn_gain - hat_Ft_Fn_loss  
#        hat_cFn = - hat_Fn_Ft_loss - hat_Fn_St_loss  
#
#        prod_list = [None] * (len(self.prim_vars) - 2)
#        prod_list[0] = hat_nSh  # -df.conditional(df.le(nF, 0.3), 1.0, 0.0) * df.Constant(0.1) * nSt / 2.0
#        prod_list[1] = hat_nSt  # - df.conditional(df.le(nF, 0.3), 1.0, 0.0) * df.Constant(0.1) * nSt / 2.0
#        prod_list[2] = hat_nSn  # df.conditional(df.le(nF, 0.3), 1.0, 0.0) * df.Constant(0.1) * nSt
#        prod_list[3] = hat_cFt  # hat_cFt
#        prod_list[4] = hat_cFn  # hat_cFn
#        return prod_list
