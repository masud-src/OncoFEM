"""
In this first tutorial, the general structure elements of OncoFEM are shown. The user sets up a study with related 
subjects. Also a measurement state is set up, that could be used as an input file for further investigation. For 
simplification this example will cut at that point and shows also the possibility of OncoFEM in implementing academic 
examples. Therefore, the central classes geometry and problem are presented. Both are necessary classes, that the gather 
information. Finally, a simple quarter circle geometry is run with a stationary poisson equation.

Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""
import oncofem as of
import datetime
from tutorial import academic_geometries
"""
First step for performing investigations with OncoFEM in general is setting up a study. A study is initialized with a 
title. The initialization of the study will generate a folder for all derived interim outputs and final results. The 
base path for the studies folder can be set in config.ini file. Therefore, the first thing that should be done is 
changing the STUDIES_DIR to the desired directory. If not already done in the installation process, please add now the 
desired directory in config.ini. In the following first tutorial the study "tut_01" is initialized and serves as a first 
test example.
"""
study = of.Study("tut_01")
"""
Since in a study usually more than one subject is investigated, the user than can add several subjects by the use of the 
"create_study" method. This is the most convenient way, because the subject already is attached to the study and
important variables are shared.
"""
subj_1 = study.create_subject("Subject_1")
"""
Alternatively a subject object can be created by itself and attached to the study. 
"""
subj_2 = of.Subject("Subject_2")
subj_2.study_dir = study.dir
study.subjects.append(subj_2)
"""
To every subject in general a particular state is measured and can be taken as initial point for investigation, or time 
point for fitting, or time point for validation. In the same manner, this can be done via a "create_state" method, that 
is implemented in the subject class. Since a state usually corresponds to a time point this can be an argument for the 
initialization.
"""
state_1 = subj_1.create_state("init_state", datetime.date.today())
"""
The time point is an optional argument, that in general is set to today. Again, a state can be created in particular and 
connected to a subject.
"""
state_2 = of.State("evaluation_state", datetime.date.today())
state_2.subject = subj_1
state_2.study_dir = study.dir
"""
In order to perform patient-specific investigations, the user can add now mri measurements. Therefore, the new created 
"measure" object is initiated via the "create_measure" Method by giving the respective path and modality.
"""
measure_1 = state_1.create_measure("tutorial/data/BraTS20_Training_001/BraTS20_Training_001_t1.nii", "t1")
"""
Again, this can alternatively be done via creating the measurement by itself and manually appending to the respective 
state.
"""
measure_2 = of.Measure("tutorial/data/BraTS20_Training_001/BraTS20_Training_001_t1ce.nii", "t1ce")
measure_3 = of.Measure("tutorial/data/BraTS20_Training_001/BraTS20_Training_001_t2.nii", "t2")
measure_4 = of.Measure("tutorial/data/BraTS20_Training_001/BraTS20_Training_001_flair.nii", "flair")
measure_5 = of.Measure("tutorial/data/BraTS20_Training_001/BraTS20_Training_001_seg.nii", "seg")
state_1.measures.append(measure_2)
state_1.measures.append(measure_3)
state_1.measures.append(measure_4)
state_1.measures.append(measure_5)
"""
To load a state into the modelling part of OncoFEM, the generated input is given to a MRI object and loaded therein. The 
MRI object holds functionalities for performing generalisation, tumor segmentation and white matter segmentation. These 
are documented in a separated tutorial. 
"""
mri = of.MRI(state_1)
mri.load_measures()
"""
The gold standard for brain tumour investigations relies on four different modalities (t1, t1ce, t2, flair). For further 
investigations it is necessary, to check, whether the state contains all four modalities.  
"""
print(mri.isFullModality())
"""
For simplification, OncoFEM can also handle simple academic examples, that can be created using gmsh. In this tutorial 
folder, the file "academic_geometries.py" can be found. Herein, simple examples are already implemented and can be used 
via the respective function. In order to use these, or the mri scans, as input geometry, the mesh and its boundaries can 
be stored in a geometry object.
"""
geometry = of.Geometry()
title = "2D_QuarterCircle"
der_file = study.der_dir + title
geometry.mesh, geometry.facet_function = academic_geometries.create_2D_QuarterCircle(0.01, 1.1, 1.0, 50, der_file)
geometry.dim = 2
"""
A problem can be set up. It holds all necessary information for generating  initial boundary value problems. For a first 
little example, we will set up test problem p and add the created geometry.
"""
p = of.Problem()
p.geom = geometry
"""
and add related  parameters for several categories, including: general, time, material, initial or fem. In order to give 
maximum flexibility, all categories are empty template classes and the user can fill them with arbitrary attributes. The 
problem class also holds an attribute for the desired base model and bio-chem-model. To keep it short in this first 
example, we will cut here and run a simple poisson problem on the generated problem. We will set the outer boundary to a 
particular value, set up linear continuous Lagrange elements for discretisation and write the solution to an output file 
in the problem folder. Also a production term is added, that is written in C.
"""
p.param.gen.output_file = study.sol_dir + "tut_01"
p.param.fem.ele_type = "CG"
p.param.fem.ele_order = 1
p.param.ext.outer_load = 0.0
p.param.mat.prod_term = "10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)"
p.param.mat.prod_term_degree = 2
"""
Next, a function for solving the poisson problem is given. This takes as anargument the generated problem and is than 
solved in the finite element framework of FEniCS. Therefore, in this function, also the dolfin package is imported.
"""
def solve_poisson(p: of.Problem):
    import dolfin as df
    #  Create mesh and define function space
    V = df.FunctionSpace(p.geom.mesh, "Lagrange", 1)

    # Define boundary condition
    u0 = df.Constant(p.param.ext.outer_load)
    bc = df.DirichletBC(V, u0, p.geom.facet_function, 3)

    # Define variational problem
    dx = df.Measure("dx")
    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    f = df.Expression(p.param.mat.prod_term, degree=p.param.mat.prod_term_degree)
    a = df.inner(df.grad(u), df.grad(v))*dx
    L = f*v*dx

    # Compute solution
    u = df.Function(V)
    df.solve(a == L, u, bc)

    # Save solution in VTK format
    out_file = of.io.set_output_file(p.param.gen.output_file)
    of.io.write_field2xdmf(out_file, u, "u", 0.0)
"""
In a last step this function is run. 
"""
solve_poisson(p)
"""
Keep in mind, that with this modular programming style the function can be run with several input set-ups. These input 
set-ups can consist in different problems, states or subjects, that are generated and can be looped.
"""
