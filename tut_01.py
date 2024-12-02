"""
Quick start patient-specific tutorial

In this tutorial a training data set of the BraTS2020 challenge serves as simple test case for a simulation of
patient-specific data. For a very first intro, all image data is preprocessed, so that this case should run on any
computer and it is focussed on basic functionalities of OncoFEM. In order to perform numerical simulations, a simple
two-phase model in the framework of the Theory of Porous Media is extended about a concentration equation that
represents the edema with resolved mobile cancer cells. Therefore, the governing equations read

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
import os
#
# PROBLEM SET UP
#
# First a study object is created. This study then creates a workspace on the hard drive with two subfolders 'der' and
# 'sol'. Herein, all derived  pre-processed results and final solutions are saved.
#
study = of.structure.Study("tut_01")
#
# A problem is set up, that holds all information for a numerical simulation. Herein, all data is initialised that
# relates to the geometry. This data is then saved to the 'geom' attribute
#
p = of.Problem()
#
# With the 'load_mesh' function the mesh is read and 'read_mapped_xdmf' is used to load fields that are distributed over
# the mesh, like the mobile tumour cells or different materials, like the white matter, gray matter or csf.
#
data_dir = os.getcwd() + os.sep + "data" + os.sep
fmg = of.FieldMapGenerator()
p.geom.mesh = fmg.load_mesh(data_dir + "geometry.xdmf")
p.geom.edema_distr = fmg.read_mapped_xdmf(data_dir + "edema.xdmf")
p.geom.wm_distr = fmg.read_mapped_xdmf(data_dir + "white_matter.xdmf")
p.geom.gm_distr = fmg.read_mapped_xdmf(data_dir + "gray_matter.xdmf")
p.geom.csf_distr = fmg.read_mapped_xdmf(data_dir + "csf.xdmf")
#
# In the next block the problem is filled with parameters that are assigned. These are needed by the selected model that
# will be assigned. With the 'set_output_file' command, the solution file can be set and with this, the file will have
# all necessary settings in order to contain all output fields in one file. Be aware of the chosen solver type. The
# example in the paper runs with mumps on a 128 GB nvidia cpu. Maybe this setting will not run on your computer and you
# should try 'lu' or 'gmres'.
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
p.param.fem.solver_type = "mumps"  # Try "lu" or "gmres" if error in solve routine
p.param.fem.maxIter = 20
p.param.fem.rel = 1E-7
p.param.fem.abs = 1E-8
# ADDITIONALS
# In order to average a parameter at a particular material point, the user can take advantage of the 'set_av_params()'
# command. For example, if a specific point contains 30 percent white matter, 60 percent grey matter and 10 percent
# cerebrospinal fluid and all these compartments have a different permeability, the user can average all values, also by
# giving a factor or weight.
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
# framework of the model is set, and into process models that describe the necessary processes that will lead to an 
# interplay of the mentioned compartments quantitatively. For example, the macro-scale of a tumour is described with its 
# base model. So far, a simplified two-phase model is implemented and shall demonstrate the capabilities of the Theory 
# of Porous Media in these examples. This is done, to show the derivation of the mathematical construct of that model 
# as well as it should represent a blueprint for own models, as OncoFEM is designed to implement also custom models.  
# Such a model can be basically load via its constructor in the sub-package 'base_models'. All base models inherits from
# a parent 'base_model' class and therefore, have the method 'set_param()', which passes all needed parameters of the 
# problem. Since, the TwoPhaseModel is a continuum-mechanical model that is solved within the finite element method, and
# next step is to set the function spaces.
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
# possible to either assign a simple float or a fully distribution. When the continuum-model is FEniCS based, also C++
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
# Since, the numerical solution of that model is preserved by FEniCS, in the following code lines simple Dirichlet
# boundaries are set. Of course, also Neumann boundaries are possible, for the sake of simplicity these are neglected in
# this tutorial and the method 'set_boundaries' takes 'None' as second argument. To set Neumann boundaries, the user
# need to set the actual terms of the weak forms. It is possible to access dolfin functionalities via 'of.utils.io.df'
#
df = of.utils.io.df
bc_u_0 = df.DirichletBC(model.function_space.sub(0).sub(0), 0.0, p.geom.facet_function, 1)
bc_u_1 = df.DirichletBC(model.function_space.sub(0).sub(1), 0.0, p.geom.facet_function, 1)
bc_u_2 = df.DirichletBC(model.function_space.sub(0).sub(2), 0.0, p.geom.facet_function, 1)
bc_p_0 = df.DirichletBC(model.function_space.sub(1), 0.0, p.geom.facet_function, 1)
model.set_boundaries([bc_u_0, bc_u_1, bc_u_2, bc_p_0], None)
#
# Now, heterogeneous material properties will be set with 'set_structural_parameters()'. These are already given via the 
# 'set_params()' method. The weak form of the regarded problem is set and with that the final solver can be created. The 
# initial conditions are set and the model is finally solved via the 'solve()' method. Note: The method 
# 'set_initial_conditions(p.param.init, p.param.add)' has two arguments, where one of them is called 'p.param.add'. This 
# is, due to the set up of the actual model, which is basically a two-phase model, consisting of a solid and a fluid 
# phase. In this implementation the user can add arbitrary resolved components into the fluid phase, e.g. glucose, 
# oxygen, growth factors, etc. This is useful, especially when the user wants to implement more complex 
# bio-chemical set-ups. With 'model.solve()' the simulation is started and its results can be viewed with a visualising
# tool like ParaView.
#
model.set_structural_parameters()
model.set_weak_form()
of.utils.io.df.set_log_level(30)
model.set_solver()
model.set_initial_conditions(p.param.init, p.param.add)
model.solve()
