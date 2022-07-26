# --- Project discription ----------------------------------------------
# Element type: FE implementation of the theory of porous media for a 
#               bi-phasic material with 6 components. Brain tumour model 
# ----------------------------------------------------------------------

# Imports
import time
import os
from oncofem.struct.study import Study
from oncofem.struct.problem import Problem
import dolfin
from oncofem.helper.io import set_output_file, msh2xdmf, getXDMF
import oncofem.modelling.field_map_generator.geometry as geom
import oncofem.modelling.base_model.continuum_mechanics.tpm.gamm_file as bm

#Define Study
study = Study("GAMM")

# Defining of general Problem
x = Problem()

# General infos
x.param.gen.flag_proliferation = True
x.param.gen.flag_metabolism = True
x.param.gen.flag_apop = False
x.param.gen.flag_necrosis = False
x.param.gen.flag_VEGF = False
x.param.gen.flag_defSplit = False
x.param.gen.flag_actConf = True

# Material Parameters
x.param.mat.nS_0S = 0.4
x.param.mat.nSt_0S = 0.0
x.param.mat.cIn_0S = 0.0
x.param.mat.cIt_0S = 0.0
x.param.mat.rhoShR = 1000
x.param.mat.rhoStR = 1000
x.param.mat.rhoIR = 1000
x.param.mat.molIn = 1
x.param.mat.molIt = 1
x.param.mat.molIv = 1
x.param.mat.kI = 1e-3
x.param.mat.DIn = 1e-5
x.param.mat.DIt = 1e-8
x.param.mat.DIv = 1e-12
x.param.mat.lambdaSh = 1e7
x.param.mat.lambdaSt = 1e7
x.param.mat.muSh = 1e7
x.param.mat.muSt = 1e7

# Time Parameters
x.param.time.T_end = 5184000
x.param.time.dt = 86400

# FEM Paramereters
x.param.fem.solver_param.newton.solver_type = "mumps"
x.param.fem.solver_param.newton.maxIter = 10
x.param.fem.solver_param.newton.rel = 1E-7
x.param.fem.solver_param.newton.abs = 1E-8
x.param.fem.type_u = "CG"
x.param.fem.order_u = 1
x.param.fem.type_p = "CG"
x.param.fem.order_p = 1
x.param.fem.type_cin = "CG"
x.param.fem.order_cin = 1
x.param.fem.type_cit = "CG"
x.param.fem.order_cit = 1
x.param.fem.type_civ = "CG"
x.param.fem.order_civ = 1
x.param.fem.order_I1 = 1
x.param.fem.order_I2 = 1
x.param.fem.order_I3 = 1

# Growth parameters
x.param.mat.growth.prolifWarburgFac = 0.01
x.param.mat.growth.nutrientCellsMin = 2.0e-8
x.param.mat.growth.cIn_Max = 0.4
x.param.mat.growth.cIn_min = 0.1
x.param.mat.growth.nutrientGrowthFactor = 10  # Factor of nutrient consumption
x.param.mat.growth.cIn_tresGrowthMin = 0.1  # Survival mode
x.param.mat.growth.cIn_tresVEGF = 0.6  # Nutrient min to send out VEGF
x.param.mat.growth.nT_tres = 0.5  # Tumour big enough for VEGF
x.param.mat.growth.massIt_tres = 2.38e-18  # DEPRECATED
x.param.mat.growth.vegf_max = 1.5e-13  # Maximal VEGF concentration
x.param.mat.growth.massT_tresMin = 1.0e-12  #
# _________________ Monod Necrosis __________________________________#
x.param.mat.growth.mu_It_necros = 1.6e-10
x.param.mat.growth.K_It_necros = 0.05
x.param.mat.growth.mu_T_necros = 1.6e-10
x.param.mat.growth.K_T_necros = 0.0005
# _________________ Verhulst Prolif and Apoptosis ___________________#
x.param.mat.growth.nT_max = 0.5
x.param.mat.growth.cIt2nT = 0.01  # 9.828212e-13  # Tumour big enough for phase
x.param.mat.growth.kappa_It_prolif = 1.0e-10  # 1.163278615e-11  # 3.587214612e-11#16.0e-11#12#1200#6.0e2#3#9.0e-12 	# concentration It
x.param.mat.growth.kappa_It_apop = 6.0e-15  # concentration It
x.param.mat.growth.kappa_T_prolif = 1.131261042e-8  # 6.817376197e-8#1.648002948e-15#7.89048171e-8#12#15.0e-15#1.0e-15	# Fluid/Solid T
x.param.mat.growth.kappa_T_apop = 1.5e-19  # Fluid/Solid T
# _________________ Metabolism ______________________________________#
x.param.mat.growth.alpha_In_prolif = 1e-7  # 9.484018265e-8  # 8#5.0e-10 	# Glycolysis factor
x.param.mat.growth.alpha_In_survival = 1.0e-12  # Non-glycotic factor

x.param.gen.title = "2D_CircleRectangle"
raw_path = study.raw_dir + x.param.gen.title
der_path = study.der_dir + x.param.gen.title + os.sep
geom.create_2D_quarter_circle_in_rectangle(10, 1, 3, 1.5, raw_path) # 40
x.param.gen.eval_points = [0]

x.geom.growthArea = [5]
msh2xdmf(raw_path, der_path)
x.geom.domain, x.geom.facet_function = getXDMF(der_path)
x.geom.mesh = x.geom.facet_function.mesh()
x.geom.dx = dolfin.Measure("dx", metadata={'quadrature_degree': 2})



print("Start calculation")
start = time.time()  # start time
old_model = bm.TPM_2Phase6Component_MAfLMoCOnCOtCOv_BrainTumour()
file = set_output_file(study.sol_dir + x.param.gen.title + "/TPM")
x.param.gen.output_file = file
old_model.set_param(x)
old_model.set_function_spaces()
bc_u_0 = dolfin.DirichletBC(old_model.function_space.sub(0).sub(0), 0.0, x.geom.facet_function, 4)
bc_u_1 = dolfin.DirichletBC(old_model.function_space.sub(0).sub(1), 0.0, x.geom.facet_function, 3)
bc_p_0 = dolfin.DirichletBC(old_model.function_space.sub(1), 0.0, x.geom.facet_function, 4)
bc_p_1 = dolfin.DirichletBC(old_model.function_space.sub(1), 2.0, x.geom.facet_function, 1)
old_model.set_boundaries([bc_u_0, bc_u_1, bc_p_0, bc_p_1], None)
old_model.set_initial_condition()
old_model.solve_TPM_2Phase6Component_MAfLMoCOnCOtCOv_BrainTumour()
