# --- Project description ----------------------------------------------
# Controller File for comparison of different growth descriptions
# - Rodriguez 1994 - single phase solid with kinematic growth 
# - Ricken&Bluhm 2009 - Three-phase growth from nutrient to solid phase
# - TPM
# - TPM 2#
# ----------------------------------------------------------------------
# Imports
import time

import dolfin
import ufl
import oncofem.modelling.base_model.continuum_mechanics.tpm.simple_solid_tumor as base_model
import oncofem.helper.io as io
from oncofem.struct.study import Study
from oncofem.struct.problem import Problem
import oncofem.modelling.field_map_generator.geometry as geom

import os
from dolfin import Measure, DirichletBC

# -----------------------------------------------------------------------------------------------------
def run_terzaghi(x):
    x.param.mat.hatrhoS = 0.00
    postfix = "-Terzaghi"
    file = io.set_output_file(study.sol_dir + x.param.gen.title + "/TPM" + postfix)
    x.param.gen.output_file = file
    bm.set_param(x)
    bm.set_function_spaces()
    # Set dirichlet-bcs
    bc_u_0 = DirichletBC(bm.function_space.sub(0).sub(0), 0.0, x.geom.facet_function, 1)
    bc_u_1 = DirichletBC(bm.function_space.sub(0).sub(0), 0.0, x.geom.facet_function, 4)
    bc_u_2 = DirichletBC(bm.function_space.sub(0).sub(1), 0.0, x.geom.facet_function, 3)
    bc_p = DirichletBC(bm.function_space.sub(1), 0.0, x.geom.facet_function, 3)

    _u, _p, _nS = dolfin.split(bm.test_function)
    q_init = 1e-6
    bF_magnitude = dolfin.Constant(q_init)  # Initial value of load magnitude (will be adjusted during calculation)
    bF_direction = dolfin.FacetNormal(x.geom.mesh)
    bF3 = bF_magnitude * bF_direction  # Set traction vector on boundary
    ds2 = Measure("ds", subdomain_data=x.geom.facet_function, subdomain_id=2)  # Declares and defines top side
    n_bound = ufl.inner(bF3, _u) * ds2  # Weak form of traction bc
    bm.set_boundaries([bc_u_0, bc_u_1, bc_u_2, bc_p], n_bound)
    bm.set_initial_condition()
    return bm.solve()

def run_incompressible(x):
    return bm.solve()

def run_advection(x):
    return bm.solve()

def run_diffusion(x):
    return bm.solve()

# Define Study
study = Study("physics_test")

# Defining of general problem
x = Problem()

# Define base model
bm = base_model.BaseSimpleSolidTumor()

# Material Parameters
x.param.mat.nS_0S = 0.5
x.param.mat.kF_0S = 1.0E-6
x.param.mat.gammaFR = 1.0
x.param.mat.lambdaS = 1.4E10
x.param.mat.muS = 1.4E10
x.param.mat.rhoSR = 1.0
x.param.mat.rhoFR = 1.5

# Time Parameters
x.param.time.T_end = 5
x.param.time.dt = 1

# FEM Paramereters
x.param.fem.solver_param.newton.solver_type = "mumps"
x.param.fem.solver_param.newton.maxIter = 20
x.param.fem.solver_param.newton.rel = 1E-6
x.param.fem.solver_param.newton.abs = 1E-7
x.param.fem.type_u = "CG"
x.param.fem.order_u = 2
x.param.fem.type_p = "CG"
x.param.fem.order_p = 1
x.param.fem.type_nS = "CG"
x.param.fem.order_nS = 1

# General infos
bools_b1 = [False]
domains = ["2D_CircleRectangle"]

# Main software process
if __name__ == '__main__':

    print("Start study")
    start = time.time()  # start time
    #dolfin.set_log_level(30)

    for domain in domains:
        print("Start " + str(domain) + " at time " + str(time.time()-start))

        x.param.gen.title = domain
        raw_path = study.raw_dir + x.param.gen.title
        der_path = study.der_dir + x.param.gen.title + os.sep
        if domain == "2D_CircleRectangle":
            geom.create_2D_quarter_circle_in_rectangle(10, 1, 3, 1.5, raw_path) # 40
            x.param.gen.eval_points = [0]
        if domain == "2D_QuarterRing":
            x.geom.growthArea = [5, 6]
        else:
            x.geom.growthArea = [5]
        io.msh2xdmf(raw_path, der_path)
        x.geom.domain, x.geom.facet_function = io.getXDMF(der_path)
        x.geom.mesh = x.geom.facet_function.mesh()
        x.geom.dx = Measure("dx", metadata={'quadrature_degree': 2})

        for b1 in bools_b1:
            x.param.gen.flag_defSplit = b1
            print("Start tpm22 " + " def Split " + str(x.param.gen.flag_defSplit))
            run_terzaghi(x)

    end = time.time()
    print("Elapsed time is  {}".format(end - start))