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
        cond_1 = ufl.gt(cFt, 0.2)                               
        H1 = ufl.conditional(cond_1, df.Constant(1.0), df.Constant(0.0))

        # nS is below max value but will begin with necrosis
        cond_2 = ufl.lt(nSt, 0.9)
        H2 = ufl.conditional(cond_2, df.Constant(1.0), df.Constant(0.0))

        # nS above treshold so healthy cells don't like
        cond_3 = ufl.gt(nS, 0.89)
        H3 = ufl.conditional(cond_3, df.Constant(1.0), df.Constant(0.0))

        # Cells become necrotic
        cond_4 = ufl.lt(cFn, 0.4)
        H4 = ufl.conditional(cond_4, df.Constant(1.0), df.Constant(0.0))

        # No Nutrients left
        cond_5 = ufl.gt(cFn, 0.0)
        H5 = ufl.conditional(cond_5, df.Constant(1.0), df.Constant(0.0))

        # Metabolic switch
        cond_6 = ufl.gt(nSt, 1e-2)
        nSt_gain = nSt * df.Constant(1.5e-5)
        H_6 = ufl.conditional(cond_6, nSt_gain, 1e-6)

        # Proliferation of mobile cancer cells
        hat_Ft_Fn_gain = cFt * df.Constant(3.5e7) * (1.0 - cFt / df.Constant(self.cFt_max))

        # Proliferation of tumour
        hat_St_Fn_gain = H1 * H2 * (1.0 - H4) * H_6 * (1.0 - nSt / df.Constant(0.9))

        # Metabolism
        hat_cFn = - H5 * (hat_St_Fn_gain * (1-nS/0.9) * (1-H3) * df.Constant(0.21e1) + df.Constant(0.25e-1) * (nSt + cFt) * self.rhoStR)

        # Necrosis
        nSn_gain = (df.Constant(0.4) - ufl.sign(cFn) * cFn) * df.Constant(3e-5)
        hat_Sn_gain = H4 * nSn_gain * (1.0 - nSn / df.Constant(0.9))       
        hat_Sh_loss = H4 * nSn_gain * nSh       

        hat_nSh = - H3 * hat_St_Fn_gain - hat_Sh_loss 
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