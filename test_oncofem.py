# --- Project description ----------------------------------------------
# Controller File for comparison of different growth descriptions
# - Rodriguez 1994 - single phase solid with kinematic growth 
# - Ricken&Bluhm 2009 - Three-phase growth from nutrient to solid phase
# - TPM
# - TPM 2#
# ----------------------------------------------------------------------
# Imports
import time
import oncofem.modelling.base_model.continuum_mechanics.tpm.simple_solid_tumor as base_model
from oncofem.helper.io import set_output_file, msh2xdmf, getXDMF
import oncofem.modelling.field_map_generator.geometry as geom
from oncofem.struct.study import Study
from oncofem.struct.problem import Problem
import dolfin
import os
from dolfin import Measure, DirichletBC

# ----------------------------------------------------------------------

def run_tpm(x):
    tpm = base_model.BaseSimpleSolidTumor()
    postfix = "-withSplit" if x.param.gen.flag_defSplit else "-withoutSplit"
    file = set_output_file(study.sol_dir + x.param.gen.title + "/TPM" + postfix)
    x.param.gen.output_file = file
    tpm.set_param(x)
    tpm.set_function_spaces()
    # Set dirichlet-bcs
    if x.param.gen.title == "3D_QuarterTube":
        bc_u_0 = DirichletBC(tpm.function_space.sub(0).sub(0), 0.0, x.geom.facet_function, 2)
        bc_u_1 = DirichletBC(tpm.function_space.sub(0).sub(1), 0.0, x.geom.facet_function, 1)
        bc_u_2 = DirichletBC(tpm.function_space.sub(0).sub(2), 0.0, x.geom.facet_function, 3)
        tpm.set_boundaries([bc_u_0, bc_u_1, bc_u_2], None)
    else:
        bc_u_0 = DirichletBC(tpm.function_space.sub(0).sub(1), 0.0, x.geom.facet_function, 3)
        bc_u_1 = DirichletBC(tpm.function_space.sub(0).sub(0), 0.0, x.geom.facet_function, 4)
        tpm.set_boundaries([bc_u_0, bc_u_1], None)
    tpm.set_initial_condition()
    return tpm.solve()

# Define Study
study = Study("paper_Growth_new")

# Defining of general Problem
x = Problem()

# Material Parameters
x.param.mat.nS_0S = 0.25
x.param.mat.m = 0
x.param.mat.kF_0S = 1.0E-6
x.param.mat.gammaFR = 1.0
x.param.mat.lambdaS = 1.4E6#8
x.param.mat.muS = 1.4E6
x.param.mat.rhoSR = 1.0
x.param.mat.rhoFR = 1.5#1.5
x.param.mat.hatrhoS = 0.01

# Time Parameters
x.param.time.T_end = 10
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
x.param.fem.order_I1 = 1
x.param.fem.order_I2 = 1
x.param.fem.order_I3 = 1

# General infos
bools_b1 = [False,True]
domains = ["2D_QuarterRing"]# ["3D_Quarter_Tube"]#,"2D_Circle_Sqare"]# ["2D_QuarterRing"]

# Main software process
if __name__ == '__main__':

    print("Start study")
    start = time.time()  # start time
    dolfin.set_log_level(30)

    for domain in domains:
        print("Start " + str(domain) + " at time " + str(time.time()-start))

        x.param.gen.title = domain
        raw_path = study.raw_dir + x.param.gen.title
        der_path = study.der_dir + x.param.gen.title + os.sep
        if domain == "2D_CircleSquare": 
            geom.create_2D_quarter_circle_in_square(4, 1, 2, raw_path)
            x.param.gen.eval_points = [0, 502, 847, 1062, 30, 3077, 2184, 2565]
        if domain == "2D_CircleRectangle":
            geom.create_2D_quarter_circle_in_rectangle(40, 1, 3, 1.5, raw_path) # 40
            x.param.gen.eval_points = [0, 914, 26, 1424, 1813, 2590]
        if domain == "2D_CircleRectangleUpsideDown":
            geom.create_2D_quarter_circle_in_rectangle(40, 1, 3, 1.5, raw_path)
            x.param.gen.eval_points = [0, 914, 26, 1424, 1813, 2590]
        if domain == "3D_QuarterTube": 
            geom.create_3D_quarter_tube(15, 1, 1.2, 2, 1, raw_path)#20
            x.param.gen.eval_points = [1695, 1441]
        if domain == "2D_QuarterDart":
            geom.create_2D_QuarterDart(200, 0.4, 0.5, 0.6, raw_path)
            x.param.gen.eval_points = [0, 109, 1229, 306, 10400, 512]  # 80 elements 2d cirlce 1070, 119, 2642
        if domain == "2D_QuarterRing":
            geom.create_2D_QuarterDart(20, 0.4, 0.5, 0.6, raw_path)  # 80 20
            x.param.gen.eval_points = [0, 2]# 42 205
        if domain == "2D_QuarterRing":
            x.geom.growthArea = [5, 6]
        else:
            x.geom.growthArea = [5]
        msh2xdmf(raw_path, der_path)
        x.geom.domain, x.geom.facet_function = getXDMF(der_path)
        x.geom.mesh = x.geom.facet_function.mesh()
        x.geom.dx = Measure("dx", metadata={'quadrature_degree': 2})

        for b1 in bools_b1:
            x.param.gen.flag_defSplit = b1
            print("Start tpm22 " + " def Split " + str(x.param.gen.flag_defSplit))
            run_tpm(x)
