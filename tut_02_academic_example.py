"""
Academic test geometry tutorial

In this second tutorial, a simplified quarter circle domain shall preserve the basis for further investigations. Herein,
again the set of governing equations read

 0 = div T - hatrhoF w_F
 0 = (nS)'_S + nS div x'_S - hatnS
 0 = div x'_S + div(nF w_F) - (hatrhoS / rhoSR + hatrhoF / rhoFR)
 0 = nF (cFt_m)'_S + div(nF cFt_m w_Ft) + cFt_m (div x'_S - hatrhoS / rhoS) - hatrhoFt / MFt_m

and are solved for the primary variables of the solid displacement u_S, the solid volume fraction nS, the fluid pressure
pF and the tumor cell concentration cFt. Alternatively, to the first tutorial, the coupling to the solid body of the
micro model will be activated and a swelling can be observed.

Notes:
A gmsh ascii file is created and meshed. This mesh is loaded and a geometry object is created. Nevertheless, again a
study is set up, which will set up the workspace on the hard disc. There is no need for the handling of medical image
data and therefore, the simple test case is set up by defining the problem and creating the geometry with the predefined
function 'create_Quarter_Circle'. In the following, the problem is set up by adding various parameters. Diverging from
the first tutorial the initial conditions are set via scalar homogeneous values and functions. The parameters of the
growth function need to be scaled, because of the changed problem.
"""
# Imports
import oncofem as of
import dolfin as df
########################################################################################################################
# INPUT direct in SIMULATION CORE
#
# Predefined function for mesh creation.
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

study = of.Study("tut_02")
p = of.simulation.Problem()
p.param.gen.title = "2D_QuarterCircle"
der_file = study.der_dir + p.param.gen.title
p.geom = create_Quarter_Circle(0.01, 1.0, 200, 60, der_file, True)  # mm
# general info
p.param.gen.flag_defSplit = True
p.param.gen.output_file = of.helper.io.set_output_file(study.sol_dir + p.param.gen.title + "/TPM")
# time parameters
p.param.time.T_end = 3600 * 24.0 * 50.0  # 50 d
p.param.time.output_interval = 3600 * 12.0  # 0.5 d
p.param.time.dt = 3600 * 3.0  # 3 h
# material parameters base model
p.param.mat.rhoSR = 1190.0 * 1e-9  # kg / mm^3
p.param.mat.rhoFR = 993.3 * 1e-9  # kg / mm^3
p.param.mat.gammaFR = 1.0  # leave on 1.0!
p.param.mat.R = 8.31446261815324 * 1000  # (N mm) / (mol K)
p.param.mat.Theta = 37.0 + 273.15  # K
p.param.mat.lambdaS = 0.03312  # N / mm^2
p.param.mat.muS = 0.00662  # N / mm^2
p.param.mat.kF = 5.0e-11  # mm / s
p.param.mat.healthy_brain_nS = 0.75
# FEM parameters
p.param.fem.solver_type = "lu"
p.param.fem.maxIter = 20
p.param.fem.rel = 1E-11
p.param.fem.abs = 1E-12
# ADDITIVES material parameter
molFt = 1.3e13  # kg / mol
DFt = 5.0e-3  # mm^2/s
p.param.add.prim_vars = ["cFt"]
p.param.add.ele_types = ["CG"]
p.param.add.ele_orders = [1]
p.param.add.tensor_orders = [0]
p.param.add.molFkappa = [molFt]
p.param.add.DFkappa = [DFt]
# Initiate model
model = of.simulation.base_models.TwoPhaseModel()
model.set_param(p)
model.set_function_spaces()
# initial conditions
p.param.init.uS_0S = [0.0, 0.0]
p.param.init.p_0S = 0.0
p.param.init.nS_0S = 0.75
field = df.Expression("c0*exp(-a*(pow((x[0]-x_s),2)+pow((x[1]-y_s),2)))",
                      degree=2, c0=1.0e-1, a=0.0008, x_s=0.0, y_s=0.0)  # mmol / l
cFt_0S = df.interpolate(field, model.CG1_sca)
p.param.add.cFkappa_0S = [cFt_0S]
# Bio chemical set up
bio_model = of.simulation.process_models.VerhulstKinetic()
bio_model.set_input(model.ansatz_functions)
bio_model.flag_solid = True
bio_model.speed_cFt = 2.0e7  #5.8e6  # 10e6 * mol / (m^3 s)
bio_model.speed_nS = 2.5e-7
prod_list = bio_model.get_output()
model.set_process_models(prod_list)
# Boundary conditions
bc = []
fs = model.function_space
bc.append(df.DirichletBC(fs.sub(0).sub(1), 0.0, p.geom.facets, 2))  # 2: unstruc
bc.append(df.DirichletBC(fs.sub(0).sub(0), 0.0, p.geom.facets, 3))  # 4: unstruc
bc.append(df.DirichletBC(fs.sub(1), 0.0, p.geom.facets, 4))
model.set_boundaries(bc, None)
# Set up model and begin to solve
model.set_structural_parameters()
model.set_weak_form()
df.set_log_level(30)
model.set_solver()
model.set_initial_conditions(p.param.init, p.param.add)
model.solve()
