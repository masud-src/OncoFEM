"""
Quick start tutorial

In this tutorial a training data set of the BraTS2020 challenge serves as simple test case for a simulation of 
patient-specific data. The data consists of a standard magnetic resonance image and a respective segmentation of the
tumour compartments by an expert. In this code the basic steps of creating a numerical simulation of a patient-specific
test case are summarised. In order to perform this simplified test case, a simple two-phase model in the framework of 
the Theory of Porous Media is extended about a concentration equation that represents the edemous tissue with resolved 
mobile cancer cells. Therefore the governing equations read

 0 = div T - hatrhoF w_F
 0 = (nS)'_S + nS div x'_S - hatnS
 0 = div x'_S + div(nF w_F) - (hatrhoS / rhoSR + hatrhoF / rhoFR)
 0 = nF (cFt_m)'_S + div(nF cFt_m w_Ft) + cFt_m (div x'_S - hatrhoS / rhoS) - hatrhoFt / MFt_m

and are solved for the primary variables of the solid displacement u_S, the solid volume fraction nS, the fluid pressure
pF and the tumor cell concentration cFt. It is assumed, that the initial concentration is maximal at the solid tumour 
segmentation and minimal at the outer edge of the edema. The spreading and growing of that area is then simulated.

DEFINITION OF INPUT

Therefore, in a first step the needed input needs to be defined. In order to do so, the user needs to first create a 
study. This study then creates a workspace folder with two subfolders 'der' and 'sol' for that particular test case. 

study = of.Study("tut_00")

Herein, all derived pre-processed results and final solutions are saved and can be visualised. The studies folder need 
to be set in the config.ini file. To compare the results of different subjects in a next hierarchical level a 'subject' 
needs to be created.

subj_1 = study.create_subject("Subject_1")

This subject than can have several states of measurements taken at different time points. By means of that the initial 
state is created and the taken measurement files can be defined.

state_1 = subj_1.create_state("init_state")
measure_1 = state_1.create_measure("data/BraTS/BraTS20_Training_001/BraTS20_Training_001_t1.nii.gz", "t1")
measure_2 = state_1.create_measure("data/BraTS/BraTS20_Training_001/BraTS20_Training_001_seg.nii.gz", "seg")

A measurement can be created by the path to the relative file or directory and its modality. This ensures, that the 
information is interpreted correctly.

MRI PRE-PROCESSING

From that the mri entity of OncoFEM can be set up by giving the particular state. All measurements of that state will 
be load and the most important structural MRIs (t1, t1gd, t2, flair, seg) are set. Furthermore, the affine of the first
image is safed as a general quantity in the mri entity.

mri = of.MRI(state=state_1)

First thing that is need to be evaluated, is the tumors spatial distribution and composition. Therefore, in a next step
the tumor entity is set up. Since, this is already given in the input state, this result can be set directly. To 
identify the particular compartment of the tumor (active, necrotic, edema), the command 'set_compartment_masks()' is 
executed.

mri.set_tumor_segmentation()
mri.tumor_segmentation.infer_param.output_path = measure_2.dir_act
mri.tumor_segmentation.set_compartment_masks()

Since the brain consists of different areas with varying microstructural compositions and material properties, their
spatial distributions is of interest. To get this information the so called 'white matter segmentation' is initialised.
Herein, the default approach is to separate in between the white and grey matter, and the cerebrospinal fluid. For this
task basically the fast algorithm of the software package fsl is used. Since, this algorithm works by separating the 
grey values of the image into three different compartments, the tumor will lead to failures in the resulting 
compartments. To overcome that issue the user can chose in between two different approaches. In this example, the 
'tumor_entity_weighted' approach is chosen, which is based on the already known spatial composition of the tumor. Both
approaches are discussed in tutorial 'tut_02_mri'. The user can further chose which inputs shall be used and the 
segmentation can be run with the 'run()' command. Since this step can take several time, the already performed output
files can be used from the data folder.

mri.set_wm_segmentation()
mri.wm_segmentation.tumor_handling_approach = "tumor_entity_weighted"  # mean_averaged_value"
structural_input_files = [mri.t1_dir]
mri.wm_segmentation.set_input_wm_seg(structural_input_files)
mri.wm_segmentation.run()

MODELLING

With this now, all necessary informations are gathered. This information needs to be translated into the quantities of 
the used model. Therefore, first a problem is set up, that holds all information for a numerical simulation. To 
interpret the gathered information with respect to the used model a field map generator is initiated. This entity
generates approximative fieldsquantites of the performed discontinous distributions provided by the segmentations.

p = of.Problem(mri)
p.param.gen.title = "Subject_1"
fmap = of.modelling.FieldMapGenerator(p)

Before a field can be mapped, of course first the domain is needed, where any field can be mapped on. This is done with
the following code lines. Again the user can chose between different adjustments. A deeper look will be given in 
tutorial 'tut_03_field_map_generator'. Since this part again can be very time consuming the user can chose to perform 
this step, or take the files given in the data folder.  

fmap.volume_resolution = 30
fmap.generate_geometry_file(p.mri.t1_dir)

Next step is to map the spatial distribution of the tumor compartments onto the created geometry. Since, in this example
a simplified model is used, only the edema shall be mapped. This process is one of the most time consuming, especially
with a fine mesh and it is strongly recommended to make use of the already created files in the data folder. Note, that
the mapping is performed onto the generated file in the data folder. The mesh generator can also generate different 
meshes at the same resolution and it is only ensured that both match if they were generated together. Various settings 
can also be made, which are discussed in the associated tutorial.

fmap.edema_min_value = 1.0E-13  # max concentration
fmap.edema_max_value = 9.828212E-1  # max concentration
fmap.interpolation_method = "linear"  # nearest, cubic
fmap.run_edema_mapping()
fmap.mapped_ede_file = fmap.fmap_dir + "edema.xdmf"

In order to perform the mapping of the white matter compartments, again the user can chose how to handle the tumor 
respective area. In this example the default setting is chosen, where the tumor are is assumed to be constant white
matter.

fmap.set_mixed_masks()
fmap.run_wm_mapping()

In the following code lines several parameters are set and most should be clear. Therefore, only selected functions are
explained. 

First special command is the 'BoundingBox' which is used to set surface boundary conditions. Here it is chosen to fix 
the displacements of the brainstem. With 'mark_facet' the respective facet_fuction and domain is given. Therefore it is 
ment, if different surface conditions are present, the user needs to set different bounding boxed and give them as a 
list to the 'mark_facet' command. 

b1 = of.helper.BoundingBox(fmap.dolfin_mesh, (100.0, 129.0), (115.0, 160.0), (-20.0, 10.0))
p.geom.domain, p.geom.facet_function = fmap.mark_facet([b1])

Next the evaluated spatially varying distributions need to be loaded in again. Which can be done, via the 
'read_mapped_xdmf' command. 

p.geom.edema_distr = of.helper.io.read_mapped_xdmf(fmap.mapped_ede_file)
p.geom.wm_distr = of.helper.io.read_mapped_xdmf(fmap.mapped_wm_file)
p.geom.gm_distr = of.helper.io.read_mapped_xdmf(fmap.mapped_gm_file)
p.geom.csf_distr = of.helper.io.read_mapped_xdmf(fmap.mapped_csf_file)

In order to average parameters at a particular material point, the user can take advantage of the 'set_av_params()' 
command. For example, if the particular point contains 30 percent white matter, 60 percent grey matter and 10 percent
cerebrospinal fluid and all compartments have a different material parameter, like diffusivity, it can be done with
this. Furthermore, the user has the ability to multiply this by a factor or weight.

DFt_wm = 1e-3
DFt_gm = 1e-11
DFt_csf = 1e-10
DFt_vals = [DFt_wm, DFt_gm, DFt_csf]
DFt_spat = [p.geom.wm_distr, p.geom.gm_distr, p.geom.csf_distr]
DFt_weights = [1, 1, 1]
DFt = of.helper.set_av_params(DFt_vals, DFt_spat, DFt_weights)

To perform numerical simulation from an interdisciplinary point of view biological process models of the micro-scale and 
physical models of the macro-scale are split in OncoFEM. The macro-scale of a tumor is described with its base model and
the user can chose in between implemented models. So far, the simplified two-phase model is implemented and shall 
demonstrate the capabilities of the Theory of Porous Media. OncoFEM is designed to implement also custom models. To 
show the derivation of the mathematical construct of that model as well as a blueprint to implement own models, see 
tutorial "tut_04_basemodel". Such a model can be basically load via its constructor in the submodule 'base_model' of 
the 'modelling' submodule. All base models have the method 'set_param()', that passes all needed parameters of the 
problem. Since, the TwoPhaseModel is a continuum-mechanical model that is solved within the finite element method, next
step is to set the function spaces.

model = of.modelling.base_model.TwoPhaseModel()
file = of.helper.io.set_output_file(study.sol_dir + p.param.gen.title + "/TPM")
p.param.gen.output_file = file
model.set_param(p)
model.set_function_spaces()

Since an initial boundary value problem is created, also initial conditions need to be set up. These could also be 
dependent on primary variables and therefore, it is important that the function spaces of the model are already set up.
Furthermore, the edema distribution is set as an initial condition for the concentration. Note, that it is possible to 
either assign a simple float or a fully distribution. When the continuum-model is fenics based, also C++ expressions can
be included. 

p.param.init.uS_0S = [0.0, 0.0, 0.0]
p.param.init.p_0S = 0.0
p.param.init.nS_0S = 0.4 
cFt_0S = p.geom.edema_distr
p.param.add.cFkappa_0S = [cFt_0S]

Analogous to the base model, the user can chose the underlying bio-chemical models for the processes on the micro-scale.
In the following, a Gompertzian-like growth kinetic is used. In order, to include models that are dependent on primary
variables, these need to be given to the bio-chemical model via the method 'set_prim_vars()'. Particular arguments can
be set in default in the respective class file and can be changed from this outer control file. In order to include the
bio-chemical processes the method 'return_prod_terms()' gives the microscopic processes back and the method 
'set_bio_chem_models()' of the model loads them back in.

bio_model = of.modelling.bio_chem_models.GompertzKinetic()
bio_model.set_prim_vars(model.ansatz_functions)
bio_model.max_cFt = 9.828212E-1
bio_model.speed = 0.3
prod_list = bio_model.return_prod_terms()
model.set_bio_chem_models(prod_list)

The boundary conditions are still missing. Since, the numerical solution of that model is preserved by FEniCS, in the 
following code lines simple dirichlet boundaries are set. Of course, also Neumann boundaries are possible, for the sake 
of simplicity these are neglected in this tutorial and the method 'set_boundaries' takes 'None' as second argument. 

bc_u_0 = df.DirichletBC(model.function_space.sub(0).sub(0), 0.0, p.geom.facet_function, 1)
bc_u_1 = df.DirichletBC(model.function_space.sub(0).sub(1), 0.0, p.geom.facet_function, 1)
bc_u_2 = df.DirichletBC(model.function_space.sub(0).sub(2), 0.0, p.geom.facet_function, 1)
bc_p_0 = df.DirichletBC(model.function_space.sub(1), 0.0, p.geom.facet_function, 1)
model.set_boundaries([bc_u_0, bc_u_1, bc_u_2, bc_p_0], None)

Now, finally heterogeneous material properties. These are already given via the 'set_params()' method. The weak form of
the regarded problem is set and with that the final solver can be created. The initial conditions are set and the model
is finally solved via the 'solve()' method. Note: The method 'set_initial_conditions(p.param.init, p.param.add)' has two
arguments, where one of them is called 'p.param.add'. This is, due to the set up of the actual model, which is basically
a two-phase model, consisting of a solid and a fluid phase. In this implementation the user can add arbitrary resolved
components into the fluid phase, e.g. glucose, oxygen, growth factors, etc. This is usefull, especially when the user
wants to implement more complex bio-chemical set-ups. 

model.set_heterogenities()
model.set_weak_form()
df.set_log_level(30)
model.set_solver()
model.set_initial_conditions(p.param.init, p.param.add)
model.solve() 


Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""
# Imports
import oncofem as of
from oncofem.helper.auxillaries import BoundingBox
import dolfin as df
########################################################################################################################
# INPUT
########################################################################################################################
# Set up of test case
study = of.struc.Study("tut_00")
subj_1 = study.create_subject("Subject_1")
state_1 = subj_1.create_state("init_state")
path = "data/BraTS/BraTS20_Training_001/BraTS20_Training_001_"
measure_1 = state_1.create_measure(path + "t1.nii.gz", "t1")
measure_2 = state_1.create_measure(path + "seg.nii.gz", "seg")
########################################################################################################################
# MRI PRE-PROCESSING
########################################################################################################################
# Set up of MRI unit
mri = of.mri.MRI(state=state_1)
########################################################################################################################
# Set up tumor segmentation
mri.set_tumor_segmentation()
mri.tumor_segmentation.infer_param.output_path = measure_2.dir_act
mri.tumor_segmentation.set_compartment_masks()
########################################################################################################################
# Set up white matter segmentation
run_wms = False
if run_wms:
    mri.set_wm_segmentation()
    mri.wm_segmentation.tumor_handling_approach = "tumor_entity_weighted" 
    structural_input_files = [mri.t1_dir]
    mri.wm_segmentation.set_input_wm_seg(structural_input_files)
    mri.wm_segmentation.run()
else:
    mri.wm_mask = "data/tut_00/wm.nii.gz"
    mri.gm_mask = "data/tut_00/gm.nii.gz"
    mri.csf_mask = "data/tut_00/csf.nii.gz"
    tumor_class_0 = "data/tut_00/tumor_class_pve_0.nii.gz"
    tumor_class_1 = "data/tut_00/tumor_class_pve_1.nii.gz"
    tumor_class_2 = "data/tut_00/tumor_class_pve_2.nii.gz"
    input_tumor = [tumor_class_0, tumor_class_1, tumor_class_2]
########################################################################################################################
# MODELLING
########################################################################################################################
# Set up problem and field mapping entity
p = of.struc.Problem(mri)
p.param.gen.title = "Subject_1"
fmap = of.modelling.FieldMapGenerator(p)
########################################################################################################################
# Generate geometry
run_meshing = False
if run_meshing:
    fmap.volume_resolution = 30
    fmap.generate_geometry_file(p.mri.t1_dir)
else:
    fmap.prim_mri_mod = p.mri.t1_dir
    fmap.xdmf_file = "data/tut_00/geometry.xdmf"
    fmap.dolfin_mesh = of.helper.io.load_mesh(fmap.xdmf_file)
########################################################################################################################
# Map tumor and white matter onto generated geometry
run_tumor_mapping = False
if run_tumor_mapping:
    fmap.edema_min_value = 1.0E-13  # max concentration
    fmap.edema_max_value = 9.828212E-1  # max concentration
    fmap.interpolation_method = "linear"  # nearest, cubic
    fmap.run_edema_mapping()
    fmap.mapped_ede_file = fmap.fmap_dir + "edema.xdmf"
else:
    fmap.mapped_ede_file = "data/tut_00/edema.xdmf"

fmap.set_mixed_masks()
fmap.run_wm_mapping()
########################################################################################################################
# load geometry and mapped information into problem
b1 = BoundingBox(fmap.dolfin_mesh,(100.0, 129.0),(115.0, 160.0),(-20.0, 10.0))
p.geom.domain, p.geom.facet_function = fmap.mark_facet([b1])
p.geom.mesh = fmap.dolfin_mesh
p.geom.dim = 3
p.geom.edema_distr = of.helper.io.read_mapped_xdmf(fmap.mapped_ede_file)
p.geom.wm_distr = of.helper.io.read_mapped_xdmf(fmap.mapped_wm_file)
p.geom.gm_distr = of.helper.io.read_mapped_xdmf(fmap.mapped_gm_file)
p.geom.csf_distr = of.helper.io.read_mapped_xdmf(fmap.mapped_csf_file)
########################################################################################################################
# general info
p.param.gen.flag_defSplit = True
########################################################################################################################
# time parameters
p.param.time.T_end = 120.0 * 86400
p.param.time.output_interval = 4 * 86400
p.param.time.dt = 4 * 86400
########################################################################################################################
# material parameters base model
p.param.mat.rhoSR = 1190.0
p.param.mat.rhoFR = 1993.3
p.param.mat.gammaFR = 1.0
p.param.mat.R = 8.31446261815324
p.param.mat.Theta = 37.0
p.param.mat.lambdaS = 3312.0
p.param.mat.muS = 662.0
p.param.mat.kF = 5E-13
########################################################################################################################
# FEM Paramereters
p.param.fem.solver_type = "mumps"
p.param.fem.maxIter = 20
p.param.fem.rel = 1E-7
p.param.fem.abs = 1E-8
########################################################################################################################
# ADDITIONALS
# material parameters
molFt = 2.018E13
DFt_vals = [1e-4, 1e-6, 1e-8]
DFt_spat = [p.geom.wm_distr, p.geom.gm_distr, p.geom.csf_distr]
DFt_weights = [1, 1, 1]
DFt = of.helper.auxillaries.set_av_params(DFt_vals, DFt_spat, DFt_weights)
p.param.add.prim_vars = ["cFt"]
p.param.add.ele_types = ["CG"]
p.param.add.ele_orders = [1] 
p.param.add.tensor_orders = [0]
p.param.add.molFkappa = [molFt]
p.param.add.DFkappa = [DFt]
########################################################################################################################
# Initiate model
model = of.modelling.base_model.TwoPhaseModel()
file = of.helper.io.set_output_file(study.sol_dir + p.param.gen.title + "/TPM")
p.param.gen.output_file = file
model.set_param(p)
model.set_function_spaces()
########################################################################################################################
# initial conditions
p.param.init.uS_0S = [0.0, 0.0, 0.0]
p.param.init.p_0S = 0.0
p.param.init.nS_0S = 0.4 
cFt_0S = p.geom.edema_distr
p.param.add.cFkappa_0S = [cFt_0S]
########################################################################################################################
# Bio chemical set up
bio_model = of.modelling.bio_chem_models.VerhulstKinetic()
bio_model.set_prim_vars(model.ansatz_functions)
bio_model.max_cFt = 1.0
bio_model.speed = 0.3
prod_list = bio_model.return_prod_terms()
model.set_bio_chem_models(prod_list)
########################################################################################################################
# Boundary conditions
bc_u_0 = df.DirichletBC(model.function_space.sub(0).sub(0), 0.0, p.geom.facet_function, 1)
bc_u_1 = df.DirichletBC(model.function_space.sub(0).sub(1), 0.0, p.geom.facet_function, 1)
bc_u_2 = df.DirichletBC(model.function_space.sub(0).sub(2), 0.0, p.geom.facet_function, 1)
bc_p_0 = df.DirichletBC(model.function_space.sub(1), 0.0, p.geom.facet_function, 1)
model.set_boundaries([bc_u_0, bc_u_1, bc_u_2, bc_p_0], None)
########################################################################################################################
# Set up model and begin to solve
model.set_heterogenities()
model.set_weak_form()
df.set_log_level(30)
model.set_solver()
model.set_initial_conditions(p.param.init, p.param.add)
model.solve() 
