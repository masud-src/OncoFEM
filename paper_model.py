"""
# **************************************************************************#
#                                                                           #
# === Paper model ==========================================================#
#                                                                           #
# **************************************************************************#
# File of model paper calculation
#
# Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
#
# --------------------------------------------------------------------------#
"""

# Imports
import time
import os
import dolfin as df

import oncofem.helper.io
import oncofem.struc as str
from oncofem.modelling.field_map_generator.field_map_generator import FieldMapGenerator
from oncofem.struc.problem import Problem
from oncofem.helper.io import set_output_file, getXDMF
import oncofem.modelling.field_map_generator.geometry as geom
import oncofem.modelling.base_model.glioblastoma as bm

# define study
study = str.Study("paper_model")
x = Problem()

# geometry
x.param.gen.title = "2D_CircleRectangle_intern"
x.geom.dim = 3
der_file = study.der_dir + x.param.gen.title
der_path = der_file + os.sep
x.geom.mesh, x.geom.facet_function, area_conc, area_df = geom.create_2D_QuarterCircle_Tumor(0.01, 1.0, 0.0006, 25, der_file, der_path, 1.15E-1, 1e-5)
#fmg = FieldMapGenerator(study)

################################################################################################################
# BASE MODEL
# general info
x.param.gen.flag_proliferation = False
x.param.gen.flag_metabolism = False
x.param.gen.flag_apop = False
x.param.gen.flag_necrosis = False
x.param.gen.flag_angiogenesis = False
x.param.gen.flag_defSplit = False

# time parameters
x.param.time.T_end = 1200.0  # *86400
x.param.time.output_interval = 100.0  # *86400
x.param.time.dt = 1.0  # *86400

# material parameters base model
x.param.mat.rhoShR = 500.
x.param.mat.rhoStR = 950.  # muss größer sein als Sh
x.param.mat.rhoSnR = 1000.
x.param.mat.rhoFR = 1000.
x.param.mat.gammaFR = 1000.
x.param.mat.molFt = 1.0  # 2.018E13

# spatial varying material parameters
x.param.mat.lambdaSh = 1.e7
x.param.mat.lambdaSt = 1.5e7
x.param.mat.lambdaSn = 1.e7
x.param.mat.muSh = 1.e7
x.param.mat.muSt = 1.e7
x.param.mat.muSn = 1.e7
x.param.mat.kF = 1.e-7
x.param.mat.DFt = 1.5E-5  # * 86400

# FEM Paramereters
x.param.fem.solver_param.newton.solver_type = "lu"
x.param.fem.solver_param.newton.maxIter = 10
x.param.fem.solver_param.newton.rel = 1E-7
x.param.fem.solver_param.newton.abs = 1E-8
################################################################################################################

################################################################################################################
# ADDITIONALS
# material parameters
molFn = 1.
molFv = 1.
molFa = 1.
DFn = 1.e-11
DFv = 1.e-8
DFa = 1.e-12
x.param.add.prim_vars = ["cFn", "cFv", "cFa"]
x.param.add.ele_types = ["CG", "CG", "CG"]
x.param.add.ele_orders = [1, 1, 1] 
x.param.add.tensor_orders = [0, 0, 0]
x.param.add.molFdelta = [molFn, molFv, molFa]
x.param.add.DFdelta = [DFn, DFv, DFa]
################################################################################################################
print("Start calculation")
df.set_log_level(30)
start = time.time()  # start time
model = bm.Glioblastoma()
file = set_output_file(study.sol_dir + x.param.gen.title + "/TPM")
x.param.gen.output_file = file
model.set_param(x)
model.set_function_spaces()

################################################################################################################
# initial conditions
x.param.init.uS_0S = [0.0, 0.0, 0.0]
x.param.init.p_0S = 0.0
x.param.init.nSh_0S = 0.6  
x.param.init.nSt_0S = 0.0  # fmg.read_mapped_xdmf(fmg.mapped_solid_tumor_file)
x.param.init.nSn_0S = 0.0  # fmg.read_mapped_xdmf(fmg.mapped_necrotic_file)
x.param.init.nF_0S = 0.4
V = df.FunctionSpace(x.geom.mesh, "CG", 2)
field = df.Expression(("ct0*exp(-a*(pow((x[0]-x_source),2)+pow((x[1]-y_source),2)))"), degree=2, ct0=1.15e-1, a=100, x_source=0.0, y_source=0.0)
area_cFt = df.interpolate(field, model.CG1_sca)
x.param.init.cFt_0S = area_cFt  # field #fmg.read_mapped_xdmf(init_cFt)
cFn_0S = 1.0
cFv_0S = 0.0
cFa_0S = 0.0
x.param.add.cFdelta_0S = [cFn_0S, cFv_0S, cFa_0S]

################################################################################################################
# Bio chemical set up
u, p, nSh, nSt, nSn, cFt, cFn, cFv, cFa = df.split(model.ansatz_functions)
prod_list = [None] * (len(model.prim_vars_list)-2)
kappa_Ft_proliferation = 0.0864
cFn_min_growth = 0.35
cFd_min_impact = 5E-9
cFt_threshold = 9.828212E-13
nF = 1.0 - (nSh+nSt+nSn)
rhoFt_Prod = df.Function(model.CG1_sca)
H1 = df.conditional(df.gt(cFn, cFn_min_growth), 1.0, 0.0)
#H2 = df.conditional(df.le(cFn, cFN_min_necrosis), 1.0, 0.0)
#H3 = df.conditional(df.gt(cFt, cFt_Ft2nSt), 1.0, 0.0)
H4 = df.conditional(df.ge(cFa, cFd_min_impact), 1.0, 0.0)
#H5 = df.conditional(df.ge(cFv, 0.9875 * cFv_max), 0.0, 1.0)
#H6 = df.conditional(df.gt(cFn, cFn_min_VEGF_prod), 0.0, 1.0)
#H7 = df.conditional(df.gt(cFv, cFv_angio), 1.0, 0.0)
#H8 = df.conditional(df.ge(cFv, cFv_init), 1.0, 0.0)
H9 = df.conditional(df.gt(cFa, 0.0), 1.0, 0.0)
hat_Ft_Fn_gain = cFt*df.Constant(0.00025e1)#nF * cFt * df.Constant(x.param.mat.molFt) * H1 * kappa_Ft_proliferation * (1.0 - (cFt/cFt_threshold)) * (1.0 - H4)
prod_list[3] = hat_Ft_Fn_gain
model.set_bio_chem_models(prod_list)
################################################################################################################
# Boundary conditions
# u (x,y,z), p, nSh, nSt, nSn, cIt, cIn, cIv, cIa
bc_u_0 = df.DirichletBC(model.function_space.sub(0).sub(0), 0.0, x.geom.facet_function, 2)
bc_u_1 = df.DirichletBC(model.function_space.sub(0).sub(1), 0.0, x.geom.facet_function, 1)
bc_p_0 = df.DirichletBC(model.function_space.sub(1), 0.0, x.geom.facet_function, 1)
bc_p_1 = df.DirichletBC(model.function_space.sub(1), 0.0, x.geom.facet_function, 2)
bc_cFn_1 = df.DirichletBC(model.function_space.sub(6), 1.0, x.geom.facet_function, 1)
bc_cFn_2 = df.DirichletBC(model.function_space.sub(6), 1.0, x.geom.facet_function, 2)
################################################################################################################

model.set_boundaries([bc_u_0, bc_u_1, bc_p_0, bc_p_1, bc_cFn_1, bc_cFn_2], None)
model.set_heterogenities()
model.set_weak_form()
model.set_solver()
model.set_initial_conditions(x.param.init, x.param.add)
model.solve()
