"""
Beispiel 2Phaser TPM + poisson 2D
Start bei fieldmapping
kopplung 

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
import oncofem.struc as str
from oncofem.struc.problem import Problem
from oncofem.helper.io import set_output_file
import oncofem.modelling.base_model.glioblastoma as bm
from oncofem.modelling.bio_chem_models.simple_model import SimpleModel
import academic_geometries

# define study
study = str.Study("paper_model")
x = Problem()

# geometry
x.param.gen.title = "2D_CircleRectangle"
x.geom.dim = 2
der_file = study.der_dir + x.param.gen.title
der_path = der_file + os.sep
x.geom.mesh, x.geom.facet_function, area_conc, area_df = academic_geometries.create_2D_QuarterCircle_Tumor(0.0001, 1000.0, 1.0, 0.0006, 40, der_file, der_path, 1.15E-13, 1e-5)  # 0.01 60


################################################################################################################
# BASE MODEL
# general info
x.param.gen.flag_defSplit = True

# time parameters
x.param.time.T_end = 200.0  # *86400
x.param.time.output_interval = 24.0/24.0  # *86400
x.param.time.dt = 3.0/24.0  # *86400

# material parameters base model
x.param.mat.rhoShR = 1190.0
x.param.mat.rhoStR = 1190.0  # muss größer sein als Sh
x.param.mat.rhoSnR = 1190.0
x.param.mat.rhoFR = 1993.3
x.param.mat.gammaFR = 1.0
x.param.mat.molFt = 2.018E13
x.param.mat.R = 8.31446261815324
x.param.mat.Theta = 37.0

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
x.param.fem.solver_type = "lu"
x.param.fem.maxIter = 20
x.param.fem.rel = 1E-7
x.param.fem.abs = 1E-8
################################################################################################################

################################################################################################################
# ADDITIONALS
# material parameters
molFn = 0.18
DFn = 6.6E-10 * 86400
x.param.add.prim_vars = ["cFn"]
x.param.add.ele_types = ["CG"]
x.param.add.ele_orders = [1] 
x.param.add.tensor_orders = [0]
x.param.add.molFkappa = [molFn]
x.param.add.DFkappa = [DFn]
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
field = df.Expression(("ct0*exp(-a*(pow((x[0]-x_source),2)+pow((x[1]-y_source),2)))"), degree=2, ct0=6.15e-1, a=100, x_source=0.0, y_source=0.0)  # 1.15e-1
area_cFt = df.interpolate(field, model.CG1_sca)
x.param.init.cFt_0S = area_cFt  # field #fmg.read_mapped_xdmf(init_cFt)
cFn_0S = 1.0
cFa_0S = 0.0
x.param.add.cFkappa_0S = [cFn_0S]

################################################################################################################
# Bio chemical set up
bio_model = SimpleModel(x)
bio_model.set_prim_vars(model.ansatz_functions)
bio_model.flag_proliferation = True
bio_model.flag_metabolism = True
bio_model.flag_necrosis = True
bio_model.nSt_thres_lin_ms = 5e-5
bio_model.fac_nSt_lin_ms = 1e-1
bio_model.nu_Sh_necrosis = 1e-15 * 86400
bio_model.nu_St_necrosis = 1E-15 * 86400
bio_model.nu_Ft_necrosis = 0.0 * 86400
bio_model.cFn_min_necrosis = 0.85
bio_model.nSt_max = 0.5
bio_model.cFt_max = 9.828212E-1
bio_model.cFn_min_growth = 0.35
bio_model.nu_In_basal = 8.64e-28
bio_model.nu_Ft_proliferation = 0.0864
bio_model.nu_St_proliferation = 0.35856e-3  # 0.35856
bio_model.f_proli = 8.64e-5
prod_list = bio_model.return_prod_terms()
model.set_bio_chem_models(prod_list)
################################################################################################################
# Boundary conditions
# u (x,y,z), p, nSh, nSt, nSn, cFt, cFn, cFa
bc_u_0 = df.DirichletBC(model.function_space.sub(0).sub(0), 0.0, x.geom.facet_function, 3)
bc_u_1 = df.DirichletBC(model.function_space.sub(0).sub(1), 0.0, x.geom.facet_function, 2)
bc_p_0 = df.DirichletBC(model.function_space.sub(1), 0.0, x.geom.facet_function, 4)
bc_cFn_1 = df.DirichletBC(model.function_space.sub(6), 1.0, x.geom.facet_function, 4)
################################################################################################################

model.set_boundaries([bc_u_0, bc_u_1, bc_p_0, bc_cFn_1], None)
model.set_heterogenities()
model.set_weak_form()
model.set_solver()
model.set_initial_conditions(x.param.init, x.param.add)
model.solve()
