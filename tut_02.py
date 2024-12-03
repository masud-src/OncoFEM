"""
Real medical image data

In the first tutorial, already a mesh and distributed fields have been given. In general OncoFEM is created to take real
medical image data in form of magnetic resonance images (MRI) or computer tomography (CT). In this tutorial, the same
image data is treated, but it is started by generating the mesh and map the fields from a tumour segmention. Especially,
the second computation (mapping fields into the generated mesh) is quite intense. You might want to play with the
resolution. Furthermore, some structural elements of OncoFEM are shown, that help to investigate into whole studies.
"""
import oncofem as of
import os
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
#
study = of.structure.Study("tut_02")
subj_1 = study.create_subject("Subject_1")
state_1 = subj_1.create_state("init_state")
data_dir = os.getcwd() + os.sep + "data" + os.sep
measure_1 = state_1.create_measure(data_dir + "mask.nii.gz", "mask")
#
# Before a field can be mapped, of course first the domain is needed, where any field can be mapped on. This is done
# with the following code lines. Again the user can chose between different adjustments. A deeper look will be given in
# tutorial 'tut_06_field_map_generator'. Since this part again can be very time consuming the user can chose to perform
# this step, or take the files given in the data folder.
#
p = of.Problem()
fmap = of.FieldMapGenerator(state_1.der_dir)
fmap.volume_resolution = 20
p.geom.mesh = fmap.nii2dolfin_mesh(state_1.measures["mask"])
p.geom.dim = p.geom.mesh.geometric_dimension()
#
# Again, in this tutorial the tumour will be approximated as mobile cancer cells that spread inside the tissue. In this
# tutorial the mobile cancer cells inside the edema are approximated with an interlpolation from a given mask. The mask
# is everywhere one in the brain area and two in the tumour area. The white and gray matter can be identified with the
# structure segmentation package OncoSTR. These nifti files can be interpolated analogously.
#
min = 1.0E-13  # max concentration
max = 9.828212E-1  # max concentration
ede_mask = fmap.image2mask(state_1.measures["mask"], 2)
fmap.set_affine(state_1.measures["mask"])
ede_distr = "edema_distr"
fmap.interpolate(ede_mask, ede_distr, min_value=min, max_value=max)
mapped_edema = fmap.map_field(ede_distr)
p.geom.edema_distr = fmap.read_mapped_xdmf(mapped_edema)
p.geom.wm_distr = fmap.read_mapped_xdmf(data_dir + "white_matter.xdmf")
p.geom.gm_distr = fmap.read_mapped_xdmf(data_dir + "gray_matter.xdmf")
p.geom.csf_distr = fmap.read_mapped_xdmf(data_dir + "csf.xdmf")
#
# In the next block the problem is filled with parameters that are assigned. These are needed by the selected model that
# will be assigned. With the 'set_output_file' command, the solution file can be set and with this the file will have
# all necessary settings in order to contain all output fields in one file. Be aware of the chosen solver type. The
# example in the paper runs with mumps on a 128 GB nvidia cpu. Maybe it will not run on your computer with 'mumps'.
# Please try 'lu' or 'gmres' if 'mumps' run into an error.
#
# general info
p.param.gen.flag_defSplit = True
p.param.gen.output_file = of.utils.io.set_output_file(study.sol_dir + "/solution")
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
p.param.fem.solver_type = "gmres"  # Try "mumps", "lu" or "gmres" if error in solution
p.param.fem.maxIter = 20
p.param.fem.rel = 1E-7
p.param.fem.abs = 1E-8
# ADDITIONALS
# In order to average parameters at a particular material point, the user can take advantage of the 'set_av_params()'
# command. For example, if the particular point contains 30 percent white matter, 60 percent grey matter and 10 percent
# cerebrospinal fluid and all compartments have a different material parameter, e.g. diffusion, it can be done with this
# Furthermore, the user has the ability to multiply this by a factor or weight.
molFt = 1.3e13
DFt_vals = [1e-4, 1e-6, 1e-6]
DFt_spat = [p.geom.wm_distr, p.geom.gm_distr, p.geom.csf_distr]
DFt_weights = [1, 1, 1]
DFt = of.utils.fem.set_av_params(DFt_vals, DFt_spat, DFt_weights)
p.param.add.prim_vars = ["cFt"]
p.param.add.ele_types = ["CG"]
p.param.add.ele_orders = [1]
p.param.add.tensor_orders = [0]
p.param.add.molFkappa = [molFt]
p.param.add.DFkappa = [DFt]
#
# INITIATION
#
# In OncoFEM the model is split into a base model, where all regarded compartments are described and the qualitative
# framework is set, and into process models that describe the necessary processes that will lead to an interplay of the
# mentioned compartments quantitatively. The macro-scale of a tumour is described with its base model and the user can
# choose in between implemented models. OncoFEM is designed to implement also custom models. So far, the simplified
# two-phase model is implemented and shall demonstrate the capabilities of the Theory of Porous Media in these examples.
# This is done to show the derivation of the mathematical construct of that model as well as a blueprint to implement
# own models. Such a model can be basically load via its constructor in the sub-package 'base_models'. All base models
# have the method 'set_param()', that passes all needed parameters of the problem. Since, the TwoPhaseModel is a
# continuum-mechanical model that is solved within the finite element method, next step is to set the function spaces.
#
model = of.base_models.TwoPhaseModel()
model.set_param(p)
model.set_function_spaces()
#
# Initial conditions
#
# Since an initial boundary value problem is created, also initial conditions need to be set up. These could also be
# dependent on primary variables and therefore, it is important that the function spaces of the model are already set
# up. Furthermore, the edema distribution is set as an initial condition for the concentration. Note, that it is
# possible to either assign a simple float or a fully distribution. When the continuum-model is fenics based, also C++
# expressions can be included.
#
p.param.init.uS_0S = [0.0, 0.0, 0.0]
p.param.init.p_0S = 0.0
p.param.init.nS_0S = 0.75
cFt_0S = p.geom.edema_distr
p.param.add.cFkappa_0S = [cFt_0S]
#
# Process model set up
#
# Analogous to the base model, the user can choose the process model. In the following, a Verhulst-like growth kinetic
# is used. In order, to include models that are dependent on primary variables, these need to be given via the method
# 'set_input()'. Particular arguments can be set in default in the respective class file and can be changed from this
# outer control file. In order to include the process model into the base model, the method 'get_output()' gives their
# output, that can be included with the method 'set_process_models()'.
#
bio_model = of.process_models.VerhulstKinetic()
bio_model.set_input(model.ansatz_functions)
prod_list = bio_model.get_output()
model.set_process_models(prod_list)
#
# Boundary conditions
#
# In order to assign boundaries at particular areas of the surface, it is necessary to mark these particular areas. With
# the command 'BoundingBox' such a surface is defined. Here it is chosen to fix the displacements of the brainstem. The
# 'mark_facet' command creates the respective facet_fuction and domain. Multiple bounding boxes can be implemented and
# will be named enumerating from one, so 1, 2, 3, and so on. This is important when the boundary will be set.
#
bounds = [(100.0, 129.0), (115.0, 160.0), (-20.0, 10.0)]
b1 = of.utils.fem.BoundingBox(p.geom.mesh, bounds)
p.geom.domain, p.geom.facet_function = of.utils.fem.mark_facet(p.geom.mesh, [b1])
p.geom.dim = p.geom.mesh.geometric_dimension()
#
# Since, the numerical solution of that model is preserved by FEniCS, in the following code lines simple dirichlet
# boundaries are set. Of course, also Neumann boundaries are possible, for the sake of simplicity these are neglected in
# this tutorial and the method 'set_boundaries' takes 'None' as second argument. To set Neumann boundaries, the user
# need to set the actual terms of the weak forms.
#
bc_u_0 = of.utils.io.df.DirichletBC(model.function_space.sub(0).sub(0), 0.0, p.geom.facet_function, 1)
bc_u_1 = of.utils.io.df.DirichletBC(model.function_space.sub(0).sub(1), 0.0, p.geom.facet_function, 1)
bc_u_2 = of.utils.io.df.DirichletBC(model.function_space.sub(0).sub(2), 0.0, p.geom.facet_function, 1)
bc_p_0 = of.utils.io.df.DirichletBC(model.function_space.sub(1), 0.0, p.geom.facet_function, 1)
model.set_boundaries([bc_u_0, bc_u_1, bc_u_2, bc_p_0], None)
#
# Now, finally heterogeneous material properties. These are already given via the 'set_params()' method. The weak form
# of the regarded problem is set and with that the final solver can be created. The initial conditions are set and the
# model is finally solved via the 'solve()' method. Note: The method 'set_initial_conditions(p.param.init, p.param.add)'
# has two arguments, where one of them is called 'p.param.add'. This is, due to the set up of the actual model, which is
# basically a two-phase model, consisting of a solid and a fluid phase. In this implementation the user can add
# arbitrary resolved components into the fluid phase, e.g. glucose, oxygen, growth factors, etc. This is useful,
# especially when the user wants to implement more complex bio-chemical set-ups.
#
model.set_structural_parameters()
model.set_weak_form()
of.utils.io.df.set_log_level(30)
model.set_solver()
model.set_initial_conditions(p.param.init, p.param.add)
model.solve()
