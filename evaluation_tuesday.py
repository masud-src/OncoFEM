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
import oncofem.modelling.base_model.continuum_mechanics.tpm.glioblastoma as bm

#Define Study
study = Study("GAMM")

# Defining of general Problem
x = Problem()

# General infos
x.param.gen.flag_proliferation = False
x.param.gen.flag_metabolism = False
x.param.gen.flag_apop = False
x.param.gen.flag_necrosis = False
x.param.gen.flag_VEGF = False
x.param.gen.flag_defSplit = False
x.param.gen.flag_actConf = True

# Material Parameters
x.param.mat.nS_0S = 0.4
x.param.mat.nSt_0S = 0.0
x.param.mat.rhoShR = 1000
x.param.mat.rhoStR = 1000
x.param.mat.rhoSnR = 1000
x.param.mat.rhoFR = 1000
x.param.mat.gammaFR = 1000
x.param.mat.molFn = 1
x.param.mat.molFt = 1
x.param.mat.molFv = 1
x.param.mat.molFa = 1
x.param.mat.kF = 1e-3
x.param.mat.DFn = 1e-5
x.param.mat.DFt = 1e-3
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
x.geom.dx = dolfin.Measure("dx", metadata={'quadrature_degree': 2})

print("Start calculation")
#dolfin.set_log_level(30)
start = time.time()  # start time
old_model = bm.Glioblastoma()
file = set_output_file(study.sol_dir + x.param.gen.title + "/TPM")
x.param.gen.output_file = file
old_model.set_param(x)
old_model.set_function_spaces()
########################################################
# Boundary conditions
# u (x,y,z), p, nSh, nSt, nSn, cIn, cIt, cIv, cIa
bc_u_0 = dolfin.DirichletBC(old_model.function_space.sub(0).sub(0), 0.0, x.geom.facet_function, 4)
bc_u_1 = dolfin.DirichletBC(old_model.function_space.sub(0).sub(1), 0.0, x.geom.facet_function, 3)
bc_cFn_1 = dolfin.DirichletBC(old_model.function_space.sub(5), 1e-2, x.geom.facet_function, 1)
bc_cFn_2 = dolfin.DirichletBC(old_model.function_space.sub(5), 1e-2, x.geom.facet_function, 2)
old_model.set_boundaries([bc_u_0, bc_u_1, bc_cFn_1, bc_cFn_2], None)
old_model.set_initial_condition()
old_model.solve()
