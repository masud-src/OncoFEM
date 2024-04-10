"""
First test how to import liver MRI
"""
# Imports
import oncofem as of
from oncofem.helper.fem_aux import BoundingBox, mark_facet
import dolfin as df
########################################################################################################################
# INPUT
#
study = of.Study("liver_MRI_first_test")
subj_1 = study.create_subject("Subject_1")
state_1 = subj_1.create_state("init_state")
path = "/media/marlon/data/dropbox/Dropbox/Arbeit/Uni/05_MRI/public/liver/segmentations/"
measure_1 = state_1.create_measure(path + "segmentation-0.nii", "seg")

########################################################################################################################
# MRI PRE-PROCESSING
mri = of.mri.MRI(state=state_1)
mri.set_tumor_segmentation()
mri.tumor_segmentation.seg_file = measure_1.dir_act
mri.ede_mask = of.mri.MRI.image2mask(measure_1.dir_act, 2)

mri.set_structure_segmentation()
mri.structure_segmentation.tumor_handling_approach = "tumor_entity_weighted" 
#structural_input_files = [mri.seg_dir]
#mri.structure_segmentation.set_input_structure_seg(structural_input_files)
#mri.structure_segmentation.run()

########################################################################################################################
# SIMULATION
#
p = of.simulation.Problem(mri)
fmap = of.simulation.FieldMapGenerator(mri)

fmap.volume_resolution = 20
fmap.generate_geometry_file(p.mri.tumor_segmentation.seg_file)

fmap.edema_min_value = 1.0E-13  # max concentration
fmap.edema_max_value = 9.828212E-1  # max concentration
fmap.interpolation_method = "linear"  # linear or nearest
#fmap.run_edema_mapping()
fmap.mapped_ede_file = of.helper.io.map_field(p.mri.tumor_segmentation.seg_file, fmap.fmap_dir + "edema", fmap.dolfin_mesh)
#fmap.set_mixed_masks()
#fmap.run_structure_mapping()

bounds = [(-500.0, 512.0), (-500.0, 512.0), (-500.0, 500.0)]
b1 = BoundingBox(fmap.dolfin_mesh, bounds)
p.geom.domain, p.geom.facet_function = mark_facet(fmap.dolfin_mesh, [b1])
p.geom.mesh = fmap.dolfin_mesh
p.geom.dim = fmap.dolfin_mesh.geometric_dimension()
# Next the evaluated spatially varying distributions need to be loaded in again. Which can be done, via the
# 'read_mapped_xdmf' command.
p.geom.edema_distr = of.helper.io.read_mapped_xdmf(fmap.mapped_ede_file)
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
#DFt_spat = [p.geom.wm_distr, p.geom.gm_distr, p.geom.csf_distr]
DFt_weights = [1, 1, 1]
DFt = 1e-3  # of.helper.fem_aux.set_av_params(DFt_vals, DFt_spat, DFt_weights)
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
