"""
Quick start tutorial

In this tutorial a training data set of the BraTS2020 challenge serves as simple test case for a simulation of
patient-specific data. The data consists of a standard magnetic resonance image and a respective segmentation of the
tumor compartments by an expert. In this code the basic steps of creating a numerical simulation of a patient-specific
test case are summarized. In order to perform this simplified test case, a simple two-phase model in the framework of
the Theory of Porous Media is extended about a concentration equation that represents the edema with resolved mobile
cancer cells. Therefore, the governing equations read

 0 = div T - hatrhoF w_F
 0 = (nS)'_S + nS div x'_S - hatnS
 0 = div x'_S + div(nF w_F) - (hatrhoS / rhoSR + hatrhoF / rhoFR)
 0 = nF (cFt_m)'_S + div(nF cFt_m w_Ft) + cFt_m (div x'_S - hatrhoS / rhoS) - hatrhoFt / MFt_m

and are solved for the primary variables of the solid displacement u_S, the solid volume fraction nS, the fluid pressure
pF and the tumor cell concentration cFt. It is assumed, that the initial concentration is maximal at the solid tumour
segmentation and minimal at the outer edge of the edema. The spreading and growing of that area is then simulated. Since
no displacements are triggered in this first example and the pressure at the boundaries is set to zero and no pressure
gradients will evolve, the problem can be simplified into a poisson equation with

 0 = nF (cFt_m)'_S + div(nF cFt_m w_Ft) - hatrhoFt / MFt_m .

Herein, the velocity of the mobile cancer cells reduce to its diffusive part

 nF cFt_m w_Ft = - DFt / (R Theta) grad cFt_m

with the diffusion parameter DFt, that becomes a scalar value for isotropic materials, the real gas constant R and the
temperature Theta. In this test case, the diffusion parameter varies for different microstructures (white-, grey matter
and cerebrospinal fluid) and the example shows the expected spreading of the mobile cancer cells into the preferred
growth directions.
"""
# Imports
import oncofem as of
from oncofem.helper.fem_aux import BoundingBox, mark_facet
import dolfin as df
########################################################################################################################
# INPUT
#
# In a first step an input needs to be defined. To do so, first a study object is created. This study then creates a
# workspace on the hard drive with two subfolders 'der' and 'sol'. Herein, all derived  pre-processed results and final
# solutions are saved. The parent studies folder need to be set in the config.ini file. To compare the results of
# different subjects in a next hierarchical level a 'subject' needs to be created. This subject than can have several
# states of measurements taken at different time points. By means of that the initial state is created and the taken
# measurement files can be defined. A measurement can be created by the path to the relative file or directory and its
# modality. This ensures, that the information is interpreted correctly.
study = of.Study("tut_01")
subj_1 = study.create_subject("Subject_1")
state_1 = subj_1.create_state("init_state")
path = of.ONCOFEM_DIR + "/data/tutorial/BraTS/BraTS20_Training_001/BraTS20_Training_001_"
measure_1 = state_1.create_measure(path + "t1.nii.gz", "t1")
measure_2 = state_1.create_measure(path + "seg.nii.gz", "seg")
########################################################################################################################
# MRI PRE-PROCESSING
#
# The mri entity can be set up by giving the particular state. All measurements of that state will be load and the most
# important structural MRIs (t1, t1gd, t2, flair, seg) are set. Furthermore, the affine of the first image is safed as a
# general quantity in the mri entity. First thing that is need to be evaluated, is the tumors spatial distribution and
# composition. Therefore, the tumor segmentation is set up. In this test case, this is already given via the input, this
# result can be set directly. To identify the particular compartment of the tumor (active, necrotic, edema), the command
# 'set_compartment_masks()' is executed.
mri = of.simulation.MRI(state=state_1)
# Since the brain consists of different areas with varying microstructural compositions and material properties, their
# spatial distributions is of interest. To get this information the so called 'white matter segmentation' is
# initialised. Herein, the default approach is to separate in between the white and grey matter, and the cerebrospinal
# fluid. For this task basically the fast algorithm of the software package fsl is used. Since, this algorithm works by
# separating the grey values of the image into three different compartments, the tumor will lead to failures in the
# resulting compartments. To overcome that issue the user can chose in between two different approaches. In this
# example, the 'tumor_entity_weighted' approach is chosen, which is based on the already known spatial composition of
# the tumor. Both approaches are discussed in tutorial 'tut_05_mri_structure_segmentation'. The user can further chose
# which inputs shall be used and the segmentation can be run with the 'run()' command. Since this step can take several
# time, the already performed output files from the already done calculation can be used from the data folder.
# Keep in mind, that the rest of the tutorial is build up on the tumor entity weighted approach, several adjustments
# need to be done for bias corrected approach.
mri.wm_mask = of.ONCOFEM_DIR + "/data/tutorial/tut_01/wm.nii.gz"
mri.gm_mask = of.ONCOFEM_DIR + "/data/tutorial/tut_01/gm.nii.gz"
mri.csf_mask = of.ONCOFEM_DIR + "/data/tutorial/tut_01/csf.nii.gz"
tumor_class_0 = of.ONCOFEM_DIR + "/data/tutorial/tut_01/tumor_class_pve_0.nii.gz"
tumor_class_1 = of.ONCOFEM_DIR + "/data/tutorial/tut_01/tumor_class_pve_1.nii.gz"
tumor_class_2 = of.ONCOFEM_DIR + "/data/tutorial/tut_01/tumor_class_pve_2.nii.gz"
input_tumor = [tumor_class_0, tumor_class_1, tumor_class_2]
########################################################################################################################
# SIMULATION
#
# With this now, all necessary information are gathered. This information needs to be translated into the quantities of
# the used model. Therefore, first a problem is set up, that holds all information for a numerical simulation. To
# interpret the gathered information with respect to the used model a field map generator is initiated. This entity
# generates approximate field quantities of the performed discontinuous distributions provided by the segmentations.
p = of.simulation.Problem()
fmap = of.simulation.FieldMapGenerator(mri)
# Before a field can be mapped, of course first the domain is needed, where any field can be mapped on. This is done
# with the following code lines. Again the user can chose between different adjustments. A deeper look will be given in
# tutorial 'tut_06_field_map_generator'. Since this part again can be very time consuming the user can chose to perform
# this step, or take the files given in the data folder.
switch = True
if switch:
    fmap.volume_resolution = 20
    fmap.generate_geometry_file(mri.t1_dir)
else:
    fmap.prim_mri_mod = mri.t1_dir
    fmap.xdmf_file = of.ONCOFEM_DIR + "/data/tutorial/tut_01/geometry.xdmf"
    fmap.dolfin_mesh = of.helper.io.load_mesh(fmap.xdmf_file)
# Next step is to map the spatial distribution of the tumor compartments onto the created geometry. Since, in this
# example a simplified model is used, only the edema shall be mapped. This process is one of the most time consuming,
# especially with a fine mesh and it is strongly recommended to make use of the already created files in the data
# folder. Note, that the mapping is performed onto the generated file in the data folder. It is only ensured that both
# match if they were generated together. Various settings can also be made, which are discussed in the associated
# tutorial.
if switch:
    fmap.edema_min_value = 1.0E-13  # max concentration
    fmap.edema_max_value = 9.828212E-1  # max concentration
    fmap.interpolation_method = "linear"  # linear or nearest
    fmap.run_edema_mapping()
else:
    fmap.mapped_ede_file = of.ONCOFEM_DIR + "/data/tutorial/tut_01/edema.xdmf"
# In order to perform the mapping of the white matter compartments, again the user can chose how to handle the tumor
# respective area. In this example the default setting is chosen, where the tumor are is assumed to be constant white
# matter.
fmap.set_mixed_masks()
fmap.run_structure_mapping()
# In the following code lines several parameters are set and most should be clear. Therefore, only selected functions
# are explained.
#
# First special command is the 'BoundingBox' which is used to set surface boundary conditions. Here it is chosen to fix
# the displacements of the brainstem. With 'mark_facet' the respective facet_fuction and domain is given. Therefore it
# is ment, if different surface conditions are present, the user needs to set different bounding boxed and give them as
# a list to the 'mark_facet' command.
bounds = [(100.0, 129.0), (115.0, 160.0), (-20.0, 10.0)]
b1 = BoundingBox(fmap.dolfin_mesh, bounds)
p.geom.domain, p.geom.facet_function = mark_facet(fmap.dolfin_mesh, [b1])
p.geom.mesh = fmap.dolfin_mesh
p.geom.dim = fmap.dolfin_mesh.geometric_dimension()
# Next the evaluated spatially varying distributions need to be loaded in again. Which can be done, via the
# 'read_mapped_xdmf' command.
p.geom.edema_distr = of.helper.io.read_mapped_xdmf(fmap.mapped_ede_file)
p.geom.wm_distr = of.helper.io.read_mapped_xdmf(fmap.mapped_wm_file)
p.geom.gm_distr = of.helper.io.read_mapped_xdmf(fmap.mapped_gm_file)
p.geom.csf_distr = of.helper.io.read_mapped_xdmf(fmap.mapped_csf_file)
# general info
p.param.gen.flag_defSplit = True
p.param.gen.title = "Subject_1"
file = of.helper.io.set_output_file(study.sol_dir + p.param.gen.title + "/TPM")
p.param.gen.output_file = file
# time parameters
p.param.time.T_end = 120.0 * 86400
p.param.time.output_interval = 4 * 86400
p.param.time.dt = 4 * 86400
# material parameters base model
p.param.mat.rhoSR = 1190.0
p.param.mat.rhoFR = 993.3
p.param.mat.gammaFR = 1.0
p.param.mat.R = 8.31446261815324
p.param.mat.Theta = 37.0
p.param.mat.lambdaS = 3312.0
p.param.mat.muS = 662.0
p.param.mat.kF = 5.0e-13
p.param.mat.healthy_brain_nS = 0.75
# FEM Paramereters
p.param.fem.solver_type = "mumps"
p.param.fem.maxIter = 20
p.param.fem.rel = 1E-11
p.param.fem.abs = 1E-12
# ADDITIONALS
# material parameters
#
# In order to average parameters at a particular material point, the user can take advantage of the 'set_av_params()'
# command. For example, if the particular point contains 30 percent white matter, 60 percent grey matter and 10 percent
# cerebrospinal fluid and all compartments have a different material parameter, e. g. diffusion, it can be done with
# this. Furthermore, the user has the ability to multiply this by a factor or weight.
molFt = 1.3e13
DFt_vals = [1e-4, 1e-6, 1e-6]
DFt_spat = [p.geom.wm_distr, p.geom.gm_distr, p.geom.csf_distr]
DFt_weights = [1, 1, 1]
DFt = of.helper.fem_aux.set_av_params(DFt_vals, DFt_spat, DFt_weights)
p.param.add.prim_vars = ["cFt"]
p.param.add.ele_types = ["CG"]
p.param.add.ele_orders = [1]
p.param.add.tensor_orders = [0]
p.param.add.molFkappa = [molFt]
p.param.add.DFkappa = [DFt]
# Initiate model
#
# To perform numerical simulation from an interdisciplinary point of view biological process models of the micro-scale
# and physical models of the macro-scale are split in OncoFEM. The macro-scale of a tumor is described with its base
# model and the user can chose in between implemented models. So far, the simplified two-phase model is implemented and
# shall demonstrate the capabilities of the Theory of Porous Media. OncoFEM is designed to implement also custom models.
# To show the derivation of the mathematical construct of that model as well as a blueprint to implement own models.
# Such a model can be basically load via its constructor in the sub-module 'base_models' of the 'simulation' module. All
# base models have the method 'set_param()', that passes all needed parameters of the problem. Since, the TwoPhaseModel
# is a continuum-mechanical model that is solved within the finite element method, next step is to set the function
# spaces.
model = of.simulation.base_models.TwoPhaseModel()
model.set_param(p)
model.set_function_spaces()
# Initial conditions
#
# Since an initial boundary value problem is created, also initial conditions need to be set up. These could also be
# dependent on primary variables and therefore, it is important that the function spaces of the model are already set
# up. Furthermore, the edema distribution is set as an initial condition for the concentration. Note, that it is
# possible to either assign a simple float or a fully distribution. When the continuum-model is fenics based, also C++
# expressions can be included.
p.param.init.uS_0S = [0.0, 0.0, 0.0]
p.param.init.p_0S = 0.0
p.param.init.nS_0S = 0.75
cFt_0S = p.geom.edema_distr
p.param.add.cFkappa_0S = [cFt_0S]
# Bio chemical set up
#
# Analogous to the base model, the user can chose the underlying bio-chemical models for the processes on the
# micro-scale. In the following, a Verhulst-like growth kinetic is used. In order, to include models that are dependent
# on primary variables, these need to be given to the bio-chemical model via the method 'set_prim_vars()'. Particular
# arguments can be set in default in the respective class file and can be changed from this outer control file. In order
# to include the bio-chemical processes the method 'return_prod_terms()' gives the microscopic processes back and the
# method 'set_bio_chem_models()' of the model loads them back in.
bio_model = of.simulation.process_models.VerhulstKinetic()
bio_model.set_input(model.ansatz_functions)
prod_list = bio_model.get_output()
model.set_process_models(prod_list)
# Boundary conditions
#
# The boundary conditions are still missing. Since, the numerical solution of that model is preserved by FEniCS, in the
# following code lines simple dirichlet boundaries are set. Of course, also Neumann boundaries are possible, for the
# sake of simplicity these are neglected in this tutorial and the method 'set_boundaries' takes 'None' as second
# argument. To set Neumann boundaries, the user need to set the actual terms of the weak forms.
bc_u_0 = df.DirichletBC(model.function_space.sub(0).sub(0), 0.0, p.geom.facet_function, 1)
bc_u_1 = df.DirichletBC(model.function_space.sub(0).sub(1), 0.0, p.geom.facet_function, 1)
bc_u_2 = df.DirichletBC(model.function_space.sub(0).sub(2), 0.0, p.geom.facet_function, 1)
bc_p_0 = df.DirichletBC(model.function_space.sub(1), 0.0, p.geom.facet_function, 1)
model.set_boundaries([bc_u_0, bc_u_1, bc_u_2, bc_p_0], None)
# Now, finally heterogeneous material properties. These are already given via the 'set_params()' method. The weak form
# of the regarded problem is set and with that the final solver can be created. The initial conditions are set and the
# model is finally solved via the 'solve()' method. Note: The method 'set_initial_conditions(p.param.init, p.param.add)'
# has two arguments, where one of them is called 'p.param.add'. This is, due to the set up of the actual model, which is
# basically a two-phase model, consisting of a solid and a fluid phase. In this implementation the user can add
# arbitrary resolved components into the fluid phase, e.g. glucose, oxygen, growth factors, etc. This is useful,
# especially when the user wants to implement more complex bio-chemical set-ups.
model.set_structural_parameters()
model.set_weak_form()
df.set_log_level(30)
model.set_solver()
model.set_initial_conditions(p.param.init, p.param.add)
model.solve()
