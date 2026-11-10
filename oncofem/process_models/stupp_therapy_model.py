"""
Definition of bio-chemical model set-up, that is used for simulation in model paper.
"""
import dolfin as df
import ufl
import numpy as np
from .process_model import ProcessModel


class StuppTherapyModel(ProcessModel):
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
        self.nSt_necrotic_switch = self.nSt_metabolic_switch  # 1.0e-2
        self.nuSn = self.nuSt                         # 1.0e-4
        self.nuSn_init = self.nuSt_init               # 1.0e-6
        self.treat_days_c = 5
        self.rest_days_c = 2
        self.dt_frac_c = 86400
        self.n_cycles_c = 1
        self.t_start_c = 0.0                                    # Start time of treatment
        self.t_win_c = 600                                      # length of treatment
        self.C0 = 2                                             # initial dose
        self.k = 6840.                                          # halftime decay
        self.nu_Ft_c = 5.0e2                                    # treatment factor
        self.kappa_Fa_c = 1.0e2                                 # usage factor

        self.treat_days_r = 5
        self.rest_days_r = 2
        self.dt_frac_r = 86400
        self.n_cycles_r = 1
        self.t_start_r = 0.0  # Start time of treatment
        self.t_win_r = 600  # length of treatment
        self.x0 = 0.0
        self.y0 = 0.0
        self.z0 = 0.0
        self.R = 30.0
        self.w = 3.0
        self.D0 = 2.0
        self.alpha = 0.06
        self.beta = 0.006
        self.kappa_Ft_r = 1e8
        self.kappa_St_r = 1e-6

    def set_input(self, model):
        self.prim_vars = df.split(model.ansatz_functions)
        self.rhoShR = model.rhoSkappaR[0]
        self.rhoStR = model.rhoSkappaR[1]
        self.rhoSnR = model.rhoSkappaR[2]
        self.molFt = model.molFdelta[0]
        self.molFn = model.molFdelta[1]
        self.molFa = model.molFdelta[2]
        self.dt = model.dt
        self.model_time = model.time
        self.growth_time = model.growth_time
        self.growth_ramp = model.growth_ramp
        self.DG0 = model.DG0
        self.mesh = model.mesh

    def get_output(self):
        u, p, nSh, nSt, nSn, cFt, cFn, cFa = self.prim_vars
        nS = nSh + nSt + nSn
        space_correct = 1.0  # min_volume / volume

        # chemotherapy
        chemotherapy_time = (
            "(t >= t_start && (t - t_start) < n_cycles*cycle_len "
            " && fmod(t - t_start, cycle_len) < treat_days*dt_frac "
            " && fabs(fmod(t - t_start + 0.5*dt_frac, dt_frac) - 0.5*dt_frac) < t_win) "
            "? C0 : 0"
        )
        self.cycle_len_c = (self.treat_days_c + self.rest_days_c) * self.dt_frac_c
        params = dict(t=self.model_time, C0=self.C0, k=self.k, t_start=self.t_start_c, dt_frac=self.dt_frac_c,
                      treat_days=self.treat_days_c, cycle_len=self.cycle_len_c, n_cycles=self.n_cycles_c,
                      t_win=self.t_win_c, degree=2)
        dosing_c = df.Expression(chemotherapy_time, **params)
        hat_Fa_decay = ufl.conditional(ufl.gt(cFa, 1.0e-12), 0.6931471805599453 / self.k * cFa, 0.0)

        cond_c1 = ufl.gt(cFa, 1e-4)
        cond_c2 = ufl.gt(cFt, 1e-10)
        HC1_c = ufl.conditional(cond_c1, 0.0, 1.0)
        HC2_c = ufl.conditional(ufl.And(cond_c1, cond_c2), 1.0, 0.0)
        reaction = cFa * cFt * HC2_c * space_correct
        hat_Ft_c_loss = self.nu_Ft_c * reaction
        hat_Fa_c_loss = self.kappa_Fa_c * reaction

        # radiation
        gdim = self.mesh.geometry().dim()
        dist2d = "pow(x[0]-x0,2)+pow(x[1]-y0,2)"
        dist3d = dist2d + "+pow(x[2]-z0,2)"
        dist = dist2d if gdim == 2 else dist3d
        on_time = ("(t >= t_start && (t - t_start) < n_cycles*cycle_len "
                   " && fmod(t - t_start, cycle_len) < treat_days*dt_frac "
                   " && fabs(fmod(t - t_start + 0.5*dt_frac, dt_frac) - 0.5*dt_frac) < t_win) "
                   "? 1.0 : 0.0")
        radiation_time_space = f"({on_time}) * D0 * 0.5 * (1.0 - tanh((sqrt({dist}) - R)/w))"
        self.cycle_len = (self.treat_days_r + self.rest_days_r) * self.dt_frac_r

        params = dict(t=self.model_time, D0=self.D0, x0=self.x0, y0=self.y0, R=self.R, w=self.w,
                      t_start=self.t_start_r, dt_frac=self.dt_frac_r, treat_days=self.treat_days_r,
                      cycle_len=self.cycle_len, n_cycles=self.n_cycles_r,
                      t_win=self.t_win_r, degree=2)
        if gdim == 3:
            params["z0"] = self.z0
        dosing_r = df.Expression(radiation_time_space, **params)
        nR = 1 - ufl.exp(-self.alpha * dosing_r - self.beta * dosing_r * dosing_r)
        cond_c1 = ufl.gt(nSt, 1e-10)
        cond_c2 = ufl.gt(cFt, 1e-10)
        HC1_r = ufl.conditional(cond_c1, 1.0, 0.0)
        HC2_r = ufl.conditional(cond_c2, 1.0, 0.0)
        hat_St_r_loss = HC1_r * self.kappa_St_r * nR * nSt
        hat_Ft_r_loss = HC2_r * self.kappa_Ft_r * nR * cFt


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
        # growth only where cFt > 0 (as HC2_r for radiation): negative undershoots must not grow
        hat_Ft_Fn_gain = HC2_r * cFt * df.Constant(self.nuFt) * (1.0 - cFt / df.Constant(self.cFt_max)) * space_correct

        # Proliferation of tumour
        if self.solid_growth_switch:
            hat_St_Fn_gain = H1 * (1.0 - H2) * H3 * (1.0 - nSt / (nSh + nSt + df.Constant(1.0e-12))) * space_correct
        else:
            hat_St_Fn_gain = df.Constant(0.0)

        # Metabolism
        if self.metabolism_switch:
            cFn_growth = hat_St_Fn_gain * df.Constant(self.kappa_g) * (1 - nSn / self.nS_max)
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
        nSt_pos = ufl.max_value(nSt, 0.0)
        nSn_pos = ufl.max_value(nSn, 0.0)
        hat_Sn_gain = H2 * H4 * nSt_pos / (nSt_pos + nSn_pos + df.Constant(1.0e-12)) * space_correct

        # Necrotic phase
        cond_5 = ufl.gt(hat_Sn_gain, 0.0)
        H5 = ufl.conditional(cond_5, df.Constant(0.0), df.Constant(1.0))

        if self.growth_time != self.growth_ramp:
            s = (self.model_time - self.growth_time) / (self.growth_ramp - self.growth_time)
            sc = df.conditional(df.lt(s, 0), 0.0, df.conditional(df.gt(s, 1), 1.0, s))
            a = 6.0 * sc ** 5 - 15.0 * sc ** 4 + 10.0 * sc ** 3
            hat_St_Fn_gain = hat_St_Fn_gain * a
            hat_Ft_Fn_gain = hat_Ft_Fn_gain * a
            hat_Sn_gain = hat_Sn_gain * a
            hat_cFn = hat_cFn * a

        prod_list = [None] * (len(self.prim_vars) - 2)
        prod_list[0] = - HC1_c * H5 * hat_St_Fn_gain  # hat_nSh; HC1_c as for hat_nSt, so that sum(hat_nS) = 0 under TMZ
        prod_list[1] = HC1_c * H5 * hat_St_Fn_gain - hat_Sn_gain - hat_St_r_loss    # hat_nSt
        prod_list[2] = hat_Sn_gain + hat_St_r_loss  # hat_nSn
        prod_list[3] = (HC1_c * hat_Ft_Fn_gain - hat_Ft_c_loss - hat_Ft_r_loss) * self.molFt * (1 - nS)  # hat_cFt
        prod_list[4] = hat_cFn * (1 - nS)
        prod_list[5] = (dosing_c - hat_Fa_decay - hat_Fa_c_loss) * self.molFa * (1 - nS)
        return prod_list
