"""
In this file the growth of a glioblastoma is calibrated onto the pre-surgery state
"""
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
p.geom = create_Quarter_Circle(0.01, 1.0, 300, 40, der_file, True)  # 0.01 60

################################################################################################################
# BASE MODEL
# general info
p.param.gen.flag_defSplit = True
# time parameters
p.param.time.T_end = 20.0 * 86400
p.param.time.output_interval = 1.0/24.0 * 86400
p.param.time.dt = 1.0/24.0 * 86400
# material parameters base model
p.param.mat.rhoFR = 993.3 * 1e-9  # kg / mm^3
p.param.mat.gammaFR = 1.0
p.param.mat.R = 8.31446261815324 * 1000  # (N mm) / (mol K)
p.param.mat.Theta = 37.0 + 273.15  # K
p.param.mat.healthy_brain_nS = 0.75
# spatial varying material parameters
p.param.mat.kF = 5.0e-11  # mm / s
# FEM Paramereters
p.param.fem.solver_type = "mumps"
p.param.fem.maxIter = 20
p.param.fem.rel = 1E-10
p.param.fem.abs = 1E-11
################################################################################################################
# ADDITIONALS
# material parameters solid
rhoShR = 1190.0 * 1e-9  # kg / mm^3
rhoStR = 0.01 * 1190.0 * 1e-9  # kg / mm^3
rhoSnR = 0.001 * 1190.0 * 1e-9  # kg / mm^3
lambdaSh = 0.03312  # N / mm^2
lambdaSt = 0.03312  # N / mm^2
lambdaSn = 0.03312  # N / mm^2
muSh = 0.00662  # N / mm^2
muSt = 0.00662  # N / mm^2
muSn = 0.00662  # N / mm^2
p.param.add.prim_vars_solid = ["nSh", "nSt", "nSn"]
p.param.add.ele_types_solid = ["CG", "CG", "CG"]
p.param.add.ele_orders_solid = [1, 1, 1] 
p.param.add.tensor_orders_solid = [0, 0, 0]
p.param.add.rhoSdeltaR = [rhoShR, rhoStR, rhoSnR]
p.param.add.lambdaSdelta = [lambdaSh, lambdaSt, lambdaSn]
p.param.add.muSdelta = [muSh, muSt, muSn]
# material parameters fluid
molFt = 1.3e13  # kg / mol
DFt = 5.0e2  # mm^2/s
molFn = 0.18
DFn = 6.6e3  # mm^2/s
p.param.add.prim_vars_fluid = ["cFt", "cFn"]
p.param.add.ele_types_fluid = ["CG", "CG"]
p.param.add.ele_orders_fluid = [1, 1] 
p.param.add.tensor_orders_fluid = [0, 0]
p.param.add.molFkappa = [molFt, molFn]
p.param.add.DFkappa = [DFt, DFn]
#########################################
model = of.simulation.base_models.TwoPhaseArbitraryComponents()
p.param.gen.output_file = of.helper.io.set_output_file(study.sol_dir + "/gbm_ratio_calibration")
model.set_param(p)
model.set_function_spaces()
################################################################################################################
# initial conditions
p.param.init.uS_0S = [0.0, 0.0, 0.0]
p.param.init.p_0S = 0.0
nSh_0S = 0.75 
nSt_0S = 0.0  # fmg.read_mapped_xdmf(fmg.mapped_solid_tumor_file)
nSn_0S = 0.0  # fmg.read_mapped_xdmf(fmg.mapped_necrotic_file)
p.param.add.nSdelta_0S = [nSh_0S, nSt_0S, nSn_0S]
field = df.Expression("c0*exp(-a*(pow((x[0]-x_s),2)+pow((x[1]-y_s),2)))", 
                      degree=2, c0=1.0e-1, a=0.0008, x_s=0.0, y_s=0.0)  # mmol / l
cFt_0S = df.interpolate(field, model.CG1_sca)  # field #fmg.read_mapped_xdmf(init_cFt)
cFn_0S = 1.0  # df.interpolate(field, model.CG1_sca)
p.param.add.cFkappa_0S = [cFt_0S, cFn_0S]
################################################################################################################
# Bio chemical set up
bio_model = of.simulation.process_models.GBMRatioCalibration()
bio_model.set_input(model)
bio_model.flag_proliferation = True
bio_model.flag_metabolism = True
bio_model.flag_necrosis = True
prod_list = bio_model.get_output()
model.set_process_models(prod_list)
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
