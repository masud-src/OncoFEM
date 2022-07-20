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

import os
from dolfin import Measure, DirichletBC

# ----------------------------------------------------------------------
import optifen.general


def run_tpm21(x):
    tpm21 = of.models.TPM_2Phase_MAoLMo_Growth()
    #postfix = "actConf" if x.param.gen.flag_actConf else "refConf"
    postfix = "-withSplit" if x.param.gen.flag_defSplit else "withoutSplit"
    file = of.inout.set_output_file(study.sol_dir + x.param.gen.title + "/tpm21" + postfix)
    x.param.gen.output_file = file
    tpm21.set_param(x)
    tpm21.set_function_spaces()
    # Set dirichlet-bcs
    if x.param.gen.title == "3D_QuarterTube":
        bc_u_0 = DirichletBC(tpm21.function_space.sub(0).sub(0), 0.0, x.geom.facet_function, 2)
        bc_u_1 = DirichletBC(tpm21.function_space.sub(0).sub(1), 0.0, x.geom.facet_function, 1)
        bc_u_2 = DirichletBC(tpm21.function_space.sub(0).sub(2), 0.0, x.geom.facet_function, 3)
        tpm21.set_boundaries([bc_u_0, bc_u_1, bc_u_2], None)
    else:
        bc_u_0 = DirichletBC(tpm21.function_space.sub(0).sub(1), 0.0, x.geom.facet_function, 3)
        bc_u_1 = DirichletBC(tpm21.function_space.sub(0).sub(0), 0.0, x.geom.facet_function, 4)
        tpm21.set_boundaries([bc_u_0, bc_u_1], None)
    tpm21.set_initial_condition()
    return tpm21.solve()

def run_tpm22(x):
    tpm22 = of.models.TPM_2Phase_MAoLMoMAs_Growth()
    #postfix = "actConf" if x.param.gen.flag_actConf else "refConf"
    postfix = "-withSplit" if x.param.gen.flag_defSplit else "-withoutSplit"
    file = of.inout.set_output_file(study.sol_dir + x.param.gen.title + "/TPM" + postfix)
    x.param.gen.output_file = file
    tpm22.set_param(x)
    tpm22.set_function_spaces()
    # Set dirichlet-bcs
    if x.param.gen.title == "3D_QuarterTube":
        bc_u_0 = DirichletBC(tpm22.function_space.sub(0).sub(0), 0.0, x.geom.facet_function, 2)
        bc_u_1 = DirichletBC(tpm22.function_space.sub(0).sub(1), 0.0, x.geom.facet_function, 1)
        bc_u_2 = DirichletBC(tpm22.function_space.sub(0).sub(2), 0.0, x.geom.facet_function, 3)
        tpm22.set_boundaries([bc_u_0, bc_u_1, bc_u_2], None)
    else:
        bc_u_0 = DirichletBC(tpm22.function_space.sub(0).sub(1), 0.0, x.geom.facet_function, 3)
        bc_u_1 = DirichletBC(tpm22.function_space.sub(0).sub(0), 0.0, x.geom.facet_function, 4)
        tpm22.set_boundaries([bc_u_0, bc_u_1], None)
    tpm22.set_initial_condition()
    return tpm22.solve()

def run_rickenBluhm(x):
    rickenBluhm = of.models.TPM_RickenBluhm2009()
    x.param.gen.flag_simple_constitutives = True
    x.param.gen.flag_set_hatrhoS = True
    x.param.fem.type_nN = "CG"
    x.param.fem.order_nN = 1
    x.param.mat.rhoNR = x.param.mat.rhoFR
    x.param.mat.nN_0S = 0.8
    postfix = str(x.param.gen.flag_simple_constitutives) + str(x.param.gen.flag_set_hatrhoS)
    file = of.inout.set_output_file(study.sol_dir + x.param.gen.title + "/rickenbluhm-")# + postfix)
    x.param.gen.output_file = file
    rickenBluhm.set_param(x)
    rickenBluhm.set_function_spaces()
    # Set dirichlet-bcs
    bc_u_0 = DirichletBC(rickenBluhm.function_space.sub(0).sub(1), 0.0, x.geom.facet_function, 3)
    bc_u_1 = DirichletBC(rickenBluhm.function_space.sub(0).sub(0), 0.0, x.geom.facet_function, 4)
    rickenBluhm.set_boundaries([bc_u_0, bc_u_1], None)
    rickenBluhm.set_initial_condition()
    rickenBluhm.solve_TPM_RickenBluhm2009()

def run_rodriguez(x):
    rodriguez94 = of.models.Rodriguez1994()
    postfix = "actConf" if x.param.gen.flag_actConf else "refConf"
    file = of.inout.set_output_file(study.sol_dir + x.param.gen.title + "/Rodriguez")#-" + postfix)
    x.param.gen.output_file = file
    rodriguez94.set_param(x)
    rodriguez94.set_function_spaces()
    # Set dirichlet-bcs
    if x.param.gen.title == "3D_QuarterTube":
        bc_u_0 = DirichletBC(rodriguez94.function_space.sub(0), 0.0, x.geom.facet_function, 2)
        bc_u_1 = DirichletBC(rodriguez94.function_space.sub(1), 0.0, x.geom.facet_function, 1)
        bc_u_2 = DirichletBC(rodriguez94.function_space.sub(2), 0.0, x.geom.facet_function, 3)
        rodriguez94.set_boundaries([bc_u_0, bc_u_1, bc_u_2], None)
    elif x.param.gen.title == "2D_CircleRectangleUpsideDown":
        bc_u_0 = DirichletBC(rodriguez94.function_space.sub(1), 0.0, x.geom.facet_function, 2)
        bc_u_1 = DirichletBC(rodriguez94.function_space.sub(0), 0.0, x.geom.facet_function, 4)
        rodriguez94.set_boundaries([bc_u_0, bc_u_1], None)
    else:
        bc_u_0 = DirichletBC(rodriguez94.function_space.sub(1), 0.0, x.geom.facet_function, 3)
        bc_u_1 = DirichletBC(rodriguez94.function_space.sub(0), 0.0, x.geom.facet_function, 4)
        rodriguez94.set_boundaries([bc_u_0, bc_u_1], None)
    rodriguez94.set_initial_condition()
    rodriguez94.solve()


# Define Study
study = of.Study("paper_Growth")

# Defining of general Problem
x = of.Problem()

# Material Parameters
x.param.mat.nS_0S = 0.25
x.param.mat.m = 0
x.param.mat.kF_0S = 1.0E-6
x.param.mat.gammaFR = 1.0
x.param.mat.lambdaS = 1.4E6#8
x.param.mat.muS = 1.4E6
x.param.mat.rhoSR = 1.0
x.param.mat.rhoFR = 1.5#1.5
x.param.mat.hatrhoS = 0.005


# Time Parameters
x.param.time.T_end = 100
x.param.time.dt = 5

# FEM Paramereters
x.param.fem.solver_param.newton.solver_type = "superlu"
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
bools_b1 = [False]#, False]
bools_b2 = [False,True]
domains = ["3D_QuarterTube"]# ["3D_Quarter_Tube"]#,"2D_Circle_Sqare"]# ["2D_QuarterRing"]

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
            of.geom.create_2D_quarter_circle_in_square(4, 1, 2, raw_path)
            x.param.gen.eval_points = [0, 502, 847, 1062, 30, 3077, 2184, 2565]
        if domain == "2D_CircleRectangle":
            of.geom.create_2D_quarter_circle_in_rectangle(40, 1, 3, 1.5, raw_path) # 40
            x.param.gen.eval_points = [0, 914, 26, 1424, 1813, 2590]
        if domain == "2D_CircleRectangleUpsideDown":
            of.geom.create_2D_quarter_circle_in_rectangle(40, 1, 3, 1.5, raw_path)
            x.param.gen.eval_points = [0, 914, 26, 1424, 1813, 2590]
        if domain == "3D_QuarterTube": 
            of.geom.create_3D_quarter_tube(15, 1, 1.2, 2, 1, raw_path)#20
            x.param.gen.eval_points = [1695, 1441]
        if domain == "2D_QuarterDart":
            of.geom.create_2D_QuarterDart(200, 0.4, 0.5, 0.6, raw_path)
            x.param.gen.eval_points = [0, 109, 1229, 306, 10400, 512]  # 80 elements 2d cirlce 1070, 119, 2642
        if domain == "2D_QuarterRing":
            of.geom.create_2D_QuarterDart(80, 0.4, 0.5, 0.6, raw_path)  # 20
            x.param.gen.eval_points = [42, 205]
        if domain == "2D_QuarterRing":
            x.geom.growthArea = [5, 6]
        else:
            x.geom.growthArea = [5]
        of.inout.msh2xdmf(raw_path, der_path)
        x.geom.domain, x.geom.facet_function = of.inout.getXDMF(der_path)
        x.geom.mesh = x.geom.facet_function.mesh()
        x.geom.dx = Measure("dx", metadata={'quadrature_degree': 2})

        for b1 in bools_b1:
            x.param.gen.flag_actConf = b1
            for b2 in bools_b2:
                x.param.gen.flag_defSplit = b2

                print("Start tpm22 " + str(x.param.gen.flag_actConf) + " " + str(x.param.gen.flag_defSplit))
                run_tpm22(x)

            print("Start rodriguez")
            x.param.fem.solver_param.newton.solver_type = "cg"
            run_rodriguez(x)
            x.param.fem.solver_param.newton.solver_type = "superlu"


        data = optifen.inout.get_data_from_txt_files(study.sol_dir + domain)

        field_list = ["nS", "p", "u_av", "vonMises"]
        eval_points = x.param.gen.eval_points

        #field_data = optifen.inout.get_data_from_txt(field, point, data)

        for point in eval_points:
            for field in field_list:
                plot = optifen.inout.TimePlot(field + "-" + str(point), study.sol_dir + domain, False)
                plot.font_size = 20
                plot.plot_legend = False
                plot.x_label = r"time [s]"
                if field=="u_av": plot.y_label = r"$\| \textrm{\textbf{u}} \|$ [m]"
                if field=="nS": plot.y_label = r"n$^{\mathrm{S}}$ [-]"
                if field=="vonMises": plot.y_label = r"$\sigma_{\textrm{VM}}$ [N/m\textsuperscript{2}]"
                if field=="p": plot.y_label = r"$\lambda$ [N/m\textsuperscript{2}]"
                for graph in data:
                    if graph.dim!=0:
                        max_dir = graph.dim
                    else:
                        max_dir = 1
                    for dir in range(max_dir):
                        if graph.field == field and graph.point == point and graph.direction == dir:
                            graph.line_width = 1
                            graph.line_color = "black"
                            if "TPM,withSplit" in graph.label:
                                graph.line_style = "-"
                                #graph.line_marker = ""
                            elif "TPM,withoutSplit" in graph.label:
                                graph.line_style = "--"
                                #graph.line_marker = "^"
                            elif "Rodriguez" in graph.label:
                                graph.line_style = "-"
                                graph.line_marker = "^"
                            plot.data.append(graph)
                plot.plot_data()

    end = time.time()
    print("Elapsed time is  {}".format(end - start))
