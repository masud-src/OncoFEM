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
from oncofem.struc.study import Study
from oncofem.struc.problem import Problem
from oncofem.helper.io import set_output_file
import oncofem.modelling.field_map_generator.geometry as geom
import oncofem.modelling.base_model.glioblastoma as bm

# define study
study = Study("paper_model")
x = Problem()

# geometry
x.param.gen.title = "2D_CircleRectangle_intern"
x.geom.dim = 3
der_file = study.der_dir + x.param.gen.title
der_path = der_file + os.sep
x.geom.mesh, x.geom.facet_function, area_conc, area_df = geom.create_2D_QuarterCircle_Tumor(0.01, 1.0, 0.0006, 25, der_file, der_path, 1e-1)

################################################################################
# BASE MODEL
# general infos
x.param.gen.flag_proliferation = False
x.param.gen.flag_metabolism = False
x.param.gen.flag_apop = False
x.param.gen.flag_necrosis = False
x.param.gen.flag_angiogenesis = False
x.param.gen.flag_defSplit = False

# time parameters
x.param.time.T_end = 100.0
x.param.time.output_interval = 1.0
x.param.time.dt = 1.

# material parameters base model
x.param.mat.rhoShR = 500.
x.param.mat.rhoStR = 950.  # muss größer sein als Sh
x.param.mat.rhoSnR = 1000.
x.param.mat.rhoFR = 1000.
x.param.mat.gammaFR = 1000.
x.param.mat.molFt = 1.

# spatial varying material parameters
x.param.mat.lambdaSh = 1.e7
x.param.mat.lambdaSt = 1.5e7
x.param.mat.lambdaSn = 1.e7
x.param.mat.muSh = 1.e7
x.param.mat.muSt = 1.e7
x.param.mat.muSn = 1.e7
x.param.mat.kF = 1.e-7
x.param.mat.DFt = 0.5e-3

# FEM Paramereters
x.param.fem.solver_param.newton.solver_type = "mumps"
x.param.fem.solver_param.newton.maxIter = 10
x.param.fem.solver_param.newton.rel = 1E-7
x.param.fem.solver_param.newton.abs = 1E-8
################################################################################

################################################################################
# ADDITIONALS
# material parameters
molFn = 1.
molFv = 1.
molFa = 1.
DFn = 1.e-11
DFv = 1.e-8
DFa = area_df
x.param.add.prim_vars = ["cFn", "cFv", "cFa"]
x.param.add.ele_types = ["CG", "CG", "CG"]
x.param.add.ele_orders = [1, 1, 1] 
x.param.add.tensor_orders = [0, 0, 0]
x.param.add.molFdelta = [molFn, molFv, molFa]
x.param.add.DFdelta = [DFn, DFv, DFa]

# initial conditions
x.param.init.uS_0S = [0.0, 0.0, 0.0]
x.param.init.p_0S = 0.0
x.param.init.nSh_0S = 0.6  
x.param.init.nSt_0S = 0.0  # fmg.read_mapped_xdmf(fmg.mapped_solid_tumor_file)
x.param.init.nSn_0S = 0.0  # fmg.read_mapped_xdmf(fmg.mapped_necrotic_file)
x.param.init.nF_0S = 0.4
x.param.init.cFt_0S = area_conc  # fmg.read_mapped_xdmf(fmg.mapped_edema_file)
cFn_0S = 0.1
cFv_0S = 0.0
cFa_0S = 0.0
x.param.add.cFdelta_0S = [cFn_0S, cFv_0S, cFa_0S]

print("Start calculation")
df.set_log_level(30)
start = time.time()  # start time
old_model = bm.Glioblastoma()
file = set_output_file(study.sol_dir + x.param.gen.title + "/TPM")
x.param.gen.output_file = file
old_model.set_param(x)
old_model.set_function_spaces()

########################################################
# Boundary conditions
# u (x,y,z), p, nSh, nSt, nSn, cIn, cIt, cIv, cIa
bc_u_0 = df.DirichletBC(old_model.function_space.sub(0).sub(0), 0.0, x.geom.facet_function, 2)
bc_u_1 = df.DirichletBC(old_model.function_space.sub(0).sub(1), 0.0, x.geom.facet_function, 1)
bc_cFn_1 = df.DirichletBC(old_model.function_space.sub(5), 1.0, x.geom.facet_function, 1)
bc_cFn_2 = df.DirichletBC(old_model.function_space.sub(5), 1.0, x.geom.facet_function, 2)
old_model.set_boundaries([bc_u_0, bc_u_1, bc_cFn_1, bc_cFn_2], None)
old_model.set_heterogenities()
old_model.set_weak_form()
old_model.set_solver()
old_model.set_initial_conditions()
old_model.solve()
