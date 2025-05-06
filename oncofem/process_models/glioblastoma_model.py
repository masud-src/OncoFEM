"""
Definition of bio-chemical model set-up, that is used for simulation in model paper.
"""
import dolfin as df
import ufl
from .process_model import ProcessModel


class GlioblastomaModel(ProcessModel):
    def __init__(self):
        super().__init__()
        self.solid_growth_switch = True
        self.metabolism_switch = True
        self.prim_vars = None
        self.dt = None
        self.nS_max = 0.75                                      # fixed
        self.cFt_max = 9.828212e-1                              # 10e12 * mol / m^3 
        self.cFt2nSt = self.cFt_max * 0.95                      # little less than 
        self.cFn2nSn = 0.0
        self.nSt_metabolic_switch = 1.0e-2
        self.nuSt = 9.3e-5
        self.nuSt_init = 3.1e-6
        self.nuFt = 9.3e7                                  # kappa_nSt * 10e12 * mol / m^3 
        self.kappa_g = 5.5e1
        self.kappa_St_basal = 1.7e0
        self.kappa_Ft_basal = 0.0e-20
        self.nSt_necrotic_switch = self.nSt_metabolic_switch    # 1.0e-2
        self.nuSn = self.nuSt                         # 1.0e-4
        self.nuSn_init = self.nuSt_init               # 1.0e-6

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

        # cFt is larger than threshold and tumour begins to grow
        cond_1 = ufl.gt(cFt, self.cFt2nSt)
        H1 = ufl.conditional(cond_1, df.Constant(1.0), df.Constant(0.0))

        # Cells become necrotic
        cond_2 = ufl.lt(cFn, self.cFn2nSn)
        H2 = ufl.conditional(cond_2, df.Constant(1.0), df.Constant(0.0))

        # Metabolic switch
        cond_3 = ufl.gt(nSt, self.nSt_metabolic_switch)
        H3 = ufl.conditional(cond_3, nSt * df.Constant(self.nuSt), self.nuSt_init)

        # Proliferation of mobile cancer cells
        hat_Ft_Fn_gain = cFt * df.Constant(self.nuFt) * (1.0 - cFt / df.Constant(self.cFt_max))
        hat_Ft_Fn_gain = df.conditional(cFt > self.cFt_max, df.Constant(0.0), ufl.sqrt(hat_Ft_Fn_gain * hat_Ft_Fn_gain))

        # Proliferation of tumour
        if self.solid_growth_switch:
            hat_St_Fn_gain = H1 * (1.0 - H2) * H3 * (1.0 - nSt / df.Constant(self.nS_max))
        else:
            hat_St_Fn_gain = df.Constant(0.0)

        # Metabolism
        if self.metabolism_switch:
            cFn_growth = hat_St_Fn_gain * df.Constant(self.kappa_g) * (1 - nSt / self.nS_max)
            cFn_basal_cFt = df.Constant(self.kappa_Ft_basal) * cFt * self.molFt
            cFn_basal_nSt = df.Constant(self.kappa_St_basal) * nSt * self.rhoStR
            cFn_basal = cFn_basal_nSt + cFn_basal_cFt
            hat_cFn = - (1.0 - H2) * (cFn_growth + cFn_basal)
        else:
            hat_cFn = df.Constant(0.0)

        # Necrotic switch
        cond_4 = ufl.gt(nSn, self.nSt_necrotic_switch)
        nSt_gain = nSn * df.Constant(self.nuSn)
        H4 = ufl.conditional(cond_4, nSt_gain, self.nuSn_init)

        # Necrosis
        hat_Sn_gain = H2 * H4 * (1.0 - nSn / df.Constant(self.nS_max))

        # Necrotic phase
        cond_5 = ufl.gt(hat_Sn_gain, 0.0)
        H5 = ufl.conditional(cond_5, df.Constant(0.0), df.Constant(1.0))

        prod_list = [None] * (len(self.prim_vars) - 2)
        prod_list[0] = - H5 * hat_St_Fn_gain * self.dt / 3600                   # hat_nSh              
        prod_list[1] = (H5 * hat_St_Fn_gain - hat_Sn_gain) * self.dt / 3600     # hat_nSt              
        prod_list[2] = hat_Sn_gain * self.dt / 3600                             # hat_nSn              
        prod_list[3] = hat_Ft_Fn_gain * self.dt / 3600                          # hat_cFt            
        prod_list[4] = hat_cFn * self.dt / 3600              
        return prod_list
