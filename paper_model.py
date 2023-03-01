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
from oncofem.modelling.bio_chem_models.paper_Ehlers import paper_Ehlers

# define study
study = str.Study("paper_model")
x = Problem()

# geometry
x.param.gen.title = "2D_CircleRectangle_intern"
x.geom.dim = 3
der_file = study.der_dir + x.param.gen.title
der_path = der_file + os.sep
x.geom.mesh, x.geom.facet_function, area_conc, area_df = geom.create_2D_QuarterCircle_Tumor(0.01, 1.0, 0.0006, 60, der_file, der_path, 1.15E-1, 1e-5)


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
x.param.time.T_end = 120.0  # *86400
x.param.time.output_interval = 1.0  # *86400
x.param.time.dt = 1.0  # *86400

# material parameters base model
x.param.mat.rhoShR = 1190.0
x.param.mat.rhoStR = 1190.0  # muss größer sein als Sh
x.param.mat.rhoSnR = 1190.0
x.param.mat.rhoFR = 993.3
x.param.mat.gammaFR = 1.0
x.param.mat.molFt = 2.018E13

# metastatic switch
x.param.mat.cFt_ms = 7.3E-1
x.param.mat.nSt_ms = 8E-7

# spatial varying material parameters
x.param.mat.lambdaSh = 3312.0
x.param.mat.lambdaSt = 3312.0
x.param.mat.lambdaSn = 3312.0
x.param.mat.muSh = 662.0
x.param.mat.muSt = 662.0
x.param.mat.muSn = 662.0
x.param.mat.kF = 5E-13
x.param.mat.DFt = 1.5E-13 * 86400

# FEM Paramereters
x.param.fem.solver_param.newton.solver_type = "lu"
x.param.fem.solver_param.newton.maxIter = 20
x.param.fem.solver_param.newton.rel = 1E-7
x.param.fem.solver_param.newton.abs = 1E-8
################################################################################################################

################################################################################################################
# ADDITIONALS
# material parameters
molFn = 0.18
molFv = 3.8123E-2
molFa = 93.0
DFn = 6.6E-10 * 86400
DFv = 1.16E-8 * 86400
DFa = 1E-11 * 86400
x.param.add.prim_vars = ["cFn"]
x.param.add.ele_types = ["CG"]
x.param.add.ele_orders = [1] 
x.param.add.tensor_orders = [0]
x.param.add.molFdelta = [molFn]
x.param.add.DFdelta = [DFn]
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
x.param.init.nSh_0S = 0.4 
x.param.init.nSt_0S = 0.0  # fmg.read_mapped_xdmf(fmg.mapped_solid_tumor_file)
x.param.init.nSn_0S = 0.0  # fmg.read_mapped_xdmf(fmg.mapped_necrotic_file)
V = df.FunctionSpace(x.geom.mesh, "CG", 2)
field = df.Expression(("ct0*exp(-a*(pow((x[0]-x_source),2)+pow((x[1]-y_source),2)))"), degree=2, ct0=1.15e-1, a=100, x_source=0.0, y_source=0.0)
area_cFt = df.interpolate(field, model.CG1_sca)
x.param.init.cFt_0S = area_cFt  # field #fmg.read_mapped_xdmf(init_cFt)
cFn_0S = 1.0
cFv_0S = 0.0
cFa_0S = 0.0
x.param.add.cFdelta_0S = [cFn_0S]

################################################################################################################
# Bio chemical set up
bio_model = paper_Ehlers()
bio_model.set_prim_vars(model.ansatz_functions)
prod_list = bio_model.return_prod_terms()
model.set_bio_chem_models(prod_list)
################################################################################################################
# Boundary conditions
# u (x,y,z), p, nSh, nSt, nSn, cIt, cIn, cIv, cIa
bc_u_0 = df.DirichletBC(model.function_space.sub(0).sub(0), 0.0, x.geom.facet_function, 2)
bc_u_1 = df.DirichletBC(model.function_space.sub(0).sub(1), 0.0, x.geom.facet_function, 1)
bc_p_0 = df.DirichletBC(model.function_space.sub(1), 0.0, x.geom.facet_function, 1)
bc_p_1 = df.DirichletBC(model.function_space.sub(1), 0.0, x.geom.facet_function, 2)
bc_cFn_1 = df.DirichletBC(model.function_space.sub(6), 1.0, x.geom.facet_function, 3)
################################################################################################################

model.set_boundaries([bc_u_0, bc_u_1, bc_p_0, bc_p_1], None)
model.set_heterogenities()
model.set_weak_form()
model.set_solver()
model.set_initial_conditions(x.param.init, x.param.add)
model.solve()
