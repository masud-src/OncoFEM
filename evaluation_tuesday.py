# --- Project discription ----------------------------------------------
# Element type: FE implementation of the theory of porous media for a 
#               bi-phasic material with 6 components. Brain tumour model 
# ----------------------------------------------------------------------

# Imports
import time
import os
import dolfin as df
import ufl
from oncofem.struct.study import Study
from oncofem.struct.problem import Problem
from oncofem.helper.io import set_output_file, msh2xdmf, getXDMF
import oncofem.modelling.field_map_generator.geometry as geom
import oncofem.modelling.base_model.continuum_mechanics.tpm.glioblastoma as bm

#Define Study
study = Study("GAMM")

# Defining of general Problem
x = Problem()

# General infos
x.param.gen.flag_proliferation = True
x.param.gen.flag_metabolism = True
x.param.gen.flag_apop = False
x.param.gen.flag_necrosis = True
x.param.gen.flag_angiogenesis = False
x.param.gen.flag_defSplit = True

# Material Parameters
x.param.mat.nS_0S = 0.4
x.param.mat.nSt_0S = 0.0
x.param.mat.rhoShR = 1000
x.param.mat.rhoStR = 999 #muss größer sein als Sh
x.param.mat.rhoSnR = 1000
x.param.mat.rhoFR = 1000
x.param.mat.gammaFR = 1000
x.param.mat.molFn = 1
x.param.mat.molFt = 1
x.param.mat.molFv = 1
x.param.mat.molFa = 1
x.param.mat.kF = 1e-3
x.param.mat.DFn = 1e-3
x.param.mat.DFt = 0.5e-4
x.param.mat.DFv = 1e-12
x.param.mat.DFa = 1e-12
x.param.mat.lambdaSh = 1e7
x.param.mat.lambdaSt = 1.5e7
x.param.mat.lambdaSn = 1e7
x.param.mat.muSh = 1e7
x.param.mat.muSt = 1e7
x.param.mat.muSn = 1e7

# Time Parameters
x.param.time.T_end = 100
x.param.time.dt = 2

# FEM Paramereters
x.param.fem.solver_param.newton.solver_type = "lu"
x.param.fem.solver_param.newton.maxIter = 10
x.param.fem.solver_param.newton.rel = 1E-7
x.param.fem.solver_param.newton.abs = 1E-8
x.param.fem.type_u = "CG"
x.param.fem.order_u = 1
x.param.fem.type_p = "CG"
x.param.fem.order_p = 1
x.param.fem.type_nSh = "CG"
x.param.fem.order_nSh = 1
x.param.fem.type_nSt = "CG"
x.param.fem.order_nSt = 1
x.param.fem.type_nSn = "CG"
x.param.fem.order_nSn = 1
x.param.fem.type_cFn = "CG"
x.param.fem.order_cFn = 1
x.param.fem.type_cFt = "CG"
x.param.fem.order_cFt = 1
x.param.fem.type_cFv = "CG"
x.param.fem.order_cFv = 1
x.param.fem.type_cFa = "CG"
x.param.fem.order_cFa = 1
x.param.fem.order_I1 = 1
x.param.fem.order_I2 = 1
x.param.fem.order_I3 = 1

x.param.gen.title = "2D_CircleRectangle"
raw_path = study.raw_dir + x.param.gen.title
der_path = study.der_dir + x.param.gen.title + os.sep
geom.create_2D_quarter_circle_in_rectangle(10, 1, 3, 1.5, raw_path) # 40
x.param.gen.eval_points = [0]

x.geom.growthArea = [5]
msh2xdmf(raw_path, der_path)
x.geom.domain, x.geom.facet_function = getXDMF(der_path)
x.geom.mesh = x.geom.facet_function.mesh()
x.geom.dx = df.Measure("dx", metadata={'quadrature_degree': 2})

print("Start calculation")
df.set_log_level(30)
start = time.time()  # start time
old_model = bm.Glioblastoma()
file = set_output_file(study.sol_dir + x.param.gen.title + "/TPM")
x.param.gen.output_file = file
old_model.set_param(x)
old_model.set_function_spaces()

u, p, nSh, nSt, nSn, cFn, cFt, cFv, cFa = df.split(old_model.ansatz_functions)

def linear(field, alpha):
    return field * alpha

def delta2max(field, max, dt):
    return max * dt - field

def verhulst_growth(field, kappa, max_value):
    """
    solves 
    field * kappa * (1 - field / max_value)
    """
    return field * kappa * (1 - field / max_value)

tres_cFn_survival = df.Constant(0.0008)
tres_cFn_necrosis = df.Constant(0.05)
tres_cFn_vegf = df.Constant(0.002)
tres_cFt_bulk = df.Constant(0.006)
conc_vegf_max = df.Constant(0.0015)
alpha_metabolism_cFt = df.Constant(0.5e-5)
alpha_metabolism_nSt = df.Constant(1e-2)
# Define conditions
low_cFn_survival = ufl.le(cFn, tres_cFn_survival)
high_cFn_survival = ufl.gt(cFn, tres_cFn_survival)
low_cFn_necrosis = ufl.le(cFn, tres_cFn_necrosis)
low_cFn_vegf = ufl.le(cFn, tres_cFn_vegf)
mobile_cells_larger_zero = ufl.gt(cFt, df.DOLFIN_EPS)
tumor_larger_zero = ufl.gt(cFt, df.DOLFIN_EPS)  # tumor_larger_zero = ufl.gt(cFt, df.DOLFIN_EPS)
cFt_is_bulk = ufl.gt(cFt, tres_cFt_bulk)

x.bmm.bm_model_angiogenesis = ufl.conditional(tumor_larger_zero, ufl.conditional(low_cFn_vegf, delta2max(cFv, conc_vegf_max, x.param.time.dt), 0.0), 0.0)
x.bmm.bm_model_prolif_cFt = ufl.conditional(high_cFn_survival, verhulst_growth(cFt, 1e-2, 0.008), 0.0)  # TODO condition upside down
x.bmm.bm_model_prolif_nSt = ufl.conditional(high_cFn_survival, ufl.conditional(cFt_is_bulk, 2.0, 0.0), 0.0)  # TODO condition upside down
x.bmm.bm_model_necros_nSh = - ufl.conditional(low_cFn_necrosis, 1.0, 0.0)
x.bmm.bm_model_necros_nSt = 0.0#-ufl.conditional(low_cFn_necrosis, 3.0, 0.0)
x.bmm.bm_model_necros_cFt = -ufl.conditional(low_cFn_necrosis, x.bmm.bm_model_prolif_nSt, 0.0)
x.bmm.bm_model_apopto_nSh = 0.0
x.bmm.bm_model_apopto_nSt = 0.0
x.bmm.bm_model_apopto_cFt = 0.0
x.bmm.bm_model_apopto_cFa = 0.0
bm_model_metabo_cFt = linear(cFt, alpha_metabolism_cFt)
bm_model_metabo_nSt = linear(nSt, alpha_metabolism_nSt)
x.bmm.bm_model_metabo_cFn = - bm_model_metabo_cFt - bm_model_metabo_nSt
old_model.set_bio_chem_models(x)
########################################################
# Boundary conditions
# u (x,y,z), p, nSh, nSt, nSn, cIn, cIt, cIv, cIa
bc_u_0 = df.DirichletBC(old_model.function_space.sub(0).sub(0), 0.0, x.geom.facet_function, 4)
bc_u_1 = df.DirichletBC(old_model.function_space.sub(0).sub(1), 0.0, x.geom.facet_function, 3)
bc_cFn_1 = df.DirichletBC(old_model.function_space.sub(5), 1e-1, x.geom.facet_function, 1)
bc_cFn_2 = df.DirichletBC(old_model.function_space.sub(5), 1e-1, x.geom.facet_function, 2)
old_model.set_boundaries([bc_u_0, bc_u_1, bc_cFn_1, bc_cFn_2], None)
old_model.set_initial_condition()
old_model.solve()
