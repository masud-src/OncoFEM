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

To load a state into the modelling part of OncoFEM, the generated input is given to a MRI object and loaded therein. The 
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
import oncofem as of
import dolfin as df
import datetime
from tutorial.data.academic_geometries import create_2D_QuarterCircle
"""

"""
study = of.struc.Study("tut_01")
"""

"""
subj_1 = study.create_subject("Subject_1")
"""

"""
subj_2 = of.struc.Subject("Subject_2")
subj_2.study_dir = study.dir
study.subjects.append(subj_2)
"""

"""
state_1 = subj_1.create_state("init_state", datetime.date.today())
"""

"""
state_2 = of.struc.State("evaluation_state", datetime.date(1999, 12, 20))
state_2.subject = subj_1
state_2.study_dir = study.dir
"""

"""
image_path = "data/BraTS/BraTS20_Training_001/BraTS20_Training_001_"
measure_1 = state_1.create_measure(image_path + "t1.nii.gz", "t1")
"""

"""
measure_2 = of.struc.Measure(image_path + "t1ce.nii.gz", "t1ce")
measure_3 = of.struc.Measure(image_path + "t2.nii.gz", "t2")
measure_4 = of.struc.Measure(image_path + "flair.nii.gz", "flair")
measure_5 = of.struc.Measure(image_path + "seg.nii.gz", "seg")
state_1.measures.append(measure_2)
state_1.measures.append(measure_3)
state_1.measures.append(measure_4)
state_1.measures.append(measure_5)
"""

"""
mri = of.mri.MRI(state_1)
mri.load_measures()
print(mri.isFullModality())
"""

"""
g = of.struc.Geometry()
title = "2D_QuarterCircle"
der_file = study.der_dir + title
g.mesh, g.facet_function = create_2D_QuarterCircle(0.01, 1.1, 1.0, 50, der_file)
g.dim = 2
"""

"""
p = of.struc.Problem()
p.geom = g
"""

"""
# general info
p.param.gen.output_file = study.sol_dir + "tut_01"
p.param.gen.flag_defSplit = True
# time parameters
p.param.time.T_end = 150.0
p.param.time.output_interval = 5.0
p.param.time.dt = 5.0
# material parameters base model
p.param.mat.rhoSR = 750.0
p.param.mat.rhoFR = 1000.0
p.param.mat.gammaFR = 1.0
p.param.mat.R = 8.31446261815324
p.param.mat.Theta = 37.0
p.param.mat.lambdaS = 3312.0
p.param.mat.muS = 662.0
p.param.mat.kF = 5E-13
# FEM Paramereters
p.param.fem.solver_type = "mumps"
p.param.fem.maxIter = 20
p.param.fem.rel = 1E-7
p.param.fem.abs = 1E-8
# ADDITIONALS
# material parameters
molFt = 1.0
DFt = 1e-2
p.param.add.prim_vars = ["cFt"]
p.param.add.ele_types = ["CG"]
p.param.add.ele_orders = [1] 
p.param.add.tensor_orders = [0]
p.param.add.molFkappa = [molFt]
p.param.add.DFkappa = [DFt]
# Initiate model
model = of.modelling.base_model.TwoPhaseModel()
p.param.gen.output_file = of.helper.io.set_output_file(p.param.gen.output_file + "/TPM")
model.set_param(p)
model.set_function_spaces()
# initial conditions
p.param.init.uS_0S = [0.0, 0.0]
p.param.init.p_0S = 0.0
p.param.init.nS_0S = 0.4 
field = df.Expression(("c0*exp(-a*(pow((x[0]-x_s),2)+pow((x[1]-y_s),2)))"), degree=2, c0=6.15e-1, a=100, x_s=0.0, y_s=0.0)
cFt_0S = df.interpolate(field, model.CG1_sca)
p.param.add.cFkappa_0S = [cFt_0S]
# Bio chemical set up
bio_model = of.modelling.bio_chem_models.VerhulstKinetic()
bio_model.set_prim_vars(model.ansatz_functions)
prod_list = bio_model.return_prod_terms()
model.set_bio_chem_models(prod_list)
# Boundary conditions
bc_u_0 = df.DirichletBC(model.function_space.sub(0).sub(1), 0.0, p.geom.facet_function, 1)
bc_u_1 = df.DirichletBC(model.function_space.sub(0).sub(0), 0.0, p.geom.facet_function, 2)
bc_p_0 = df.DirichletBC(model.function_space.sub(1), 0.0, p.geom.facet_function, 2)
model.set_boundaries([bc_u_0, bc_u_1, bc_p_0], None)
# Set up model and begin to solve
model.set_heterogenities()
model.set_weak_form()
df.set_log_level(30)
model.set_solver()
model.set_initial_conditions(p.param.init, p.param.add)
model.solve()
