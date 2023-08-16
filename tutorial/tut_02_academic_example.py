"""
Structure tutorial

In this second tutorial, the general structure elements of OncoFEM are shown. The user sets up a study with related 
subjects. Also a measurement state is set up, that could be used as an input file for further investigation. For 
simplification this example will cut at that point and shows also the possibility of OncoFEM in implementing academic 
examples. Therefore, the central classes geometry and problem are presented. Both are necessary classes, that the gather 
information. Finally, a simple quarter circle geometry is run with a stationary poisson equation.

First step for performing investigations with OncoFEM in general is setting up a study. A study is initialized with a 
title. The initialization of the study will generate a folder for all derived interim outputs and final results. The 
base path for the studies folder can be set in config.ini file. Therefore, the first thing that should be done is 
changing the STUDIES_DIR to the desired directory. If not already done in the installation process, please add now the 
desired directory in config.ini. In the following first tutorial the study "tut_01" is initialized and serves as a first 
test example.

Since in a study usually more than one subject is investigated, the user than can add several subjects by the use of the 
"create_study" method. This is the most convenient way, because the subject already is attached to the study and
important variables are shared.

Alternatively a subject object can be created by itself and attached to the study. 

To every subject in general a particular state is measured and can be taken as initial point for investigation, or time 
point for fitting, or time point for validation. In the same manner, this can be done via a "create_state" method, that 
is implemented in the subject class. Since a state usually corresponds to a time point this can be an argument for the 
initialization.

The time point is an optional argument, that in general is set to today. Again, a state can be created in particular and 
connected to a subject.

In order to perform patient-specific investigations, the user can add now mri measurements. Therefore, the new created 
"measure" object is initiated via the "create_measure" Method by giving the respective path and modality.

Again, this can alternatively be done via creating the measurement by itself and manually appending to the respective 
state.

To load a state into the simulation part of OncoFEM, the generated input is given to a MRI object and loaded therein. The 
MRI object holds functionalities for performing generalisation, tumor segmentation and white matter segmentation. These 
are documented in a separated tutorial. 

The gold standard for brain tumour investigations relies on four different modalities (t1, t1ce, t2, flair). For further 
investigations it is necessary, to check, whether the state contains all four modalities.  

For simplification, OncoFEM can also handle simple academic examples, that can be created using gmsh. In this tutorial 
folder, the file "academic_geometries.py" can be found. Herein, simple examples are already implemented and can be used 
via the respective function. In order to use these, or the mri scans, as input geometry, the mesh and its boundaries can 
be stored in a geometry object.

A problem can be set up. It holds all necessary information for generating  initial boundary value problems. For a first 
little example, we will set up test problem p and add the created geometry.

and add related  parameters for several categories, including: general, time, material, initial or fem. In order to give 
maximum flexibility, all categories are empty template classes and the user can fill them with arbitrary attributes. The 
problem class also holds an attribute for the desired base model and bio-chem-model. To keep it short in this first 
example, we will cut here and run a simple poisson problem on the generated problem. We will set the outer boundary to a 
particular value, set up linear continuous Lagrange elements for discretisation and write the solution to an output file 
in the problem folder. Also a production term is added, that is written in C.

Next, a function for solving the poisson problem is given. This takes as anargument the generated problem and is than 
solved in the finite element framework of FEniCS. Therefore, in this function, also the dolfin package is imported.

In a last step this function is run. 

Keep in mind, that with this modular programming style the function can be run with several input set-ups. These input 
set-ups can consist in different problems, states or subjects, that are generated and can be looped.


Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""
# Imports
import oncofem as of
import dolfin as df
########################################################################################################################
# Predefined function for mesh creation
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
########################################################################################################################
# Set up of test case
study = of.Study("tut_02")
########################################################################################################################
# Generate geometry
p = of.simulation.Problem()
p.param.gen.title = "2D_QuarterCircle"
der_file = study.der_dir + p.param.gen.title
p.geom = create_Quarter_Circle(0.01, 1.0, 200, 60, der_file, True)  # mm
########################################################################################################################
# general info
p.param.gen.flag_defSplit = True
p.param.gen.output_file = of.helper.io.set_output_file(study.sol_dir + p.param.gen.title + "/TPM")
########################################################################################################################
# time parameters
p.param.time.T_end = 1.0 * 3600 * 24.0 * 50.0  # 5 d
p.param.time.output_interval = 1.0 * 3600 * 12.0  # 1.0 d
p.param.time.dt = 1.0 * 3600 * 3.0  # 3 h
########################################################################################################################
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
########################################################################################################################
# FEM parameters
p.param.fem.solver_type = "lu"
p.param.fem.maxIter = 20
p.param.fem.rel = 1E-9
p.param.fem.abs = 1E-10
########################################################################################################################
# ADDITIVES material parameter
molFt = 1.3e13  # kg / mol
DFt = 5.0e-3  # mm^2/s
p.param.add.prim_vars = ["cFt"]
p.param.add.ele_types = ["CG"]
p.param.add.ele_orders = [1] 
p.param.add.tensor_orders = [0]
p.param.add.molFkappa = [molFt]
p.param.add.DFkappa = [DFt]
########################################################################################################################
# Initiate model
model = of.simulation.base_models.TwoPhaseModel()
model.set_param(p)
model.set_function_spaces()
########################################################################################################################
# initial conditions
p.param.init.uS_0S = [0.0, 0.0]
p.param.init.p_0S = 0.0
p.param.init.nS_0S = 0.75
field = df.Expression("c0*exp(-a*(pow((x[0]-x_s),2)+pow((x[1]-y_s),2)))", 
                      degree=2, c0=1.0e-1, a=0.0008, x_s=0.0, y_s=0.0)  # mmol / l
cFt_0S = df.interpolate(field, model.CG1_sca)
p.param.add.cFkappa_0S = [cFt_0S]
########################################################################################################################
# Bio chemical set up
bio_model = of.simulation.micro_models.VerhulstKinetic()
bio_model.set_input(model.ansatz_functions)
bio_model.flag_solid = True
prod_list = bio_model.get_output()
model.set_micro_models(prod_list)
########################################################################################################################
# Boundary conditions
bc = []
fs = model.function_space
bc.append(df.DirichletBC(fs.sub(0).sub(1), 0.0, p.geom.facets, 2))  # 2: unstruc
bc.append(df.DirichletBC(fs.sub(0).sub(0), 0.0, p.geom.facets, 3))  # 4: unstruc
bc.append(df.DirichletBC(fs.sub(1), 0.0, p.geom.facets, 4))
model.set_boundaries(bc, None)
########################################################################################################################
# Set up model and begin to solve
model.set_structural_parameters()
model.set_weak_form()
df.set_log_level(30)
model.set_solver()
model.set_initial_conditions(p.param.init, p.param.add)
model.solve()
