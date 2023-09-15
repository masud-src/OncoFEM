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
import os
import dolfin as df
import oncofem as of

def create_Quarter_Circle(esize: float, fac: float, rad: float, 
                          lay: int, dfile: str, struc_mesh=True) -> of.simulation.problem.Geometry():
    """
    creates a 2D quarter circle with three boundary conditions. 
    """
    output = of.helper.general.add_file_appendix(dfile, "geo")
    ele_size = esize * rad
    with open(output, 'w') as f:
        f.write("SetFactory(\"OpenCASCADE\");\n")
        f.write("Point(1) = {0, 0, 0, " + str(ele_size) + "};\n")
        f.write("Point(2) = {" + str(rad) + ", 0, 0, " + str(ele_size * fac) + "};\n")
        if struc_mesh:
            f.write("Line(1) = {1, 2};\n")
            f.write("Extrude {{0, 0, 1}, {0, 0, 0}, Pi/2} {\n")
            f.write("  Curve{1}; Layers{" + str(lay) + "};\n")
            f.write("}\n")
        else:
            f.write("Point(3) = {0, " + str(rad) + ", 0, " + str(ele_size * fac) + "};\n")
            f.write("Line(1) = {1, 2};\n")
            f.write("Line(2) = {3, 1};\n")
            f.write("Circle(3) = {2, 1, 3};\n")
            f.write("Curve Loop(1) = {2, 1, 3};\n")
            f.write("Surface(1) = {1};\n")
        f.write("Physical Surface(\"4\") = {1};\n")
        f.write("Physical Curve(\"1\") = {1};\n")
        f.write("Physical Curve(\"2\") = {3};\n")
        f.write("Physical Curve(\"3\") = {2};\n")
    done = of.helper.general.run_shell_command("gmsh -2 " + output)
    of.helper.io.msh2xdmf(dfile, dfile + "/", correct_gmsh=True)
    _, facet_function = of.helper.io.getXDMF(dfile + "/")
    g = of.simulation.problem.Geometry()
    g.mesh = facet_function.mesh()
    g.facets = facet_function
    g.dim = g.mesh.geometric_dimension()
    return g


# define study
study = of.Study("gbm_ratio_calibration")
p = of.simulation.Problem()

# geometry
p.param.gen.title = "2D_CircleRectangle"
der_file = study.der_dir + p.param.gen.title
p.geom = create_Quarter_Circle(0.0001, 1000.0, 1.0, 40, der_file, True)  # 0.01 60

################################################################################################################
# BASE MODEL
# general info
p.param.gen.flag_defSplit = True
# time parameters
p.param.time.T_end = 200.0  # *86400
p.param.time.output_interval = 24.0/24.0  # *86400
p.param.time.dt = 3.0/24.0  # *86400

# material parameters base model
p.param.mat.rhoShR = 1190.0
p.param.mat.rhoStR = 1190.0  # muss größer sein als Sh
p.param.mat.rhoSnR = 1190.0
p.param.mat.rhoFR = 1993.3
p.param.mat.gammaFR = 1.0
p.param.mat.molFt = 2.018E13
p.param.mat.R = 8.31446261815324
p.param.mat.Theta = 37.0
# spatial varying material parameters
p.param.mat.lambdaSh = 3312.0
p.param.mat.lambdaSt = 3312.0
p.param.mat.lambdaSn = 3312.0
p.param.mat.muSh = 662.0
p.param.mat.muSt = 662.0
p.param.mat.muSn = 662.0
p.param.mat.kF = 5E-13
p.param.mat.DFt = 1.5E-13 * 86400
# FEM Paramereters
p.param.fem.solver_type = "lu"
p.param.fem.maxIter = 20
p.param.fem.rel = 1E-7
p.param.fem.abs = 1E-8
################################################################################################################
# ADDITIONALS
# material parameters
molFn = 0.18
DFn = 6.6E-10 * 86400
p.param.add.prim_vars = ["cFn"]
p.param.add.ele_types = ["CG"]
p.param.add.ele_orders = [1] 
p.param.add.tensor_orders = [0]
p.param.add.molFkappa = [molFn]
p.param.add.DFkappa = [DFn]
#########################################
model = of.simulation.base_models.Glioblastoma()
p.param.gen.output_file = of.helper.io.set_output_file(study.sol_dir + p.param.gen.title + "/TPM")
model.set_param(p)
model.set_function_spaces()
################################################################################################################
# initial conditions
p.param.init.uS_0S = [0.0, 0.0, 0.0]
p.param.init.p_0S = 0.0
p.param.init.nSh_0S = 0.4 
p.param.init.nSt_0S = 0.0  # fmg.read_mapped_xdmf(fmg.mapped_solid_tumor_file)
p.param.init.nSn_0S = 0.0  # fmg.read_mapped_xdmf(fmg.mapped_necrotic_file)
V = df.FunctionSpace(p.geom.mesh, "CG", 2)
field = df.Expression(("ct0*exp(-a*(pow((x[0]-x_source),2)+pow((x[1]-y_source),2)))"), degree=2, ct0=6.15e-1, a=100, x_source=0.0, y_source=0.0)  # 1.15e-1
area_cFt = df.interpolate(field, model.CG1_sca)
p.param.init.cFt_0S = area_cFt  # field #fmg.read_mapped_xdmf(init_cFt)
cFn_0S = 1.0
cFa_0S = 0.0
p.param.add.cFkappa_0S = [cFn_0S]

################################################################################################################
# Bio chemical set up
bio_model = of.simulation.micro_models.GBMRatioCalibration()
bio_model.set_input(model)
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
prod_list = bio_model.get_output()
model.set_micro_models(prod_list)
################################################################################################################
# Boundary conditions
bc = []
fs = model.function_space
# u (x,y,z), p, nSh, nSt, nSn, cFt, cFn, cFa
bc.append(df.DirichletBC(model.function_space.sub(0).sub(0), 0.0, p.geom.facets, 3))
bc.append(df.DirichletBC(model.function_space.sub(0).sub(1), 0.0, p.geom.facets, 2))
bc.append(df.DirichletBC(model.function_space.sub(1), 0.0, p.geom.facets, 4))
bc.append(df.DirichletBC(model.function_space.sub(6), 1.0, p.geom.facets, 4))
################################################################################################################

model.set_boundaries(bc, None)
model.set_structural_parameters()
model.set_weak_form()
df.set_log_level(30)
model.set_solver()
model.set_initial_conditions(p.param.init, p.param.add)
model.solve()
