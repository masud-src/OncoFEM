"""
Mesh generation and mapping

In the first tutorial, already a mesh and distributed fields have been given. In general OncoFEM is created to take real
medical image data in form of magnetic resonance images (MRI) or computer tomography (CT). In this tutorial, the same
problem is treated, but it is started by generating the mesh and map the fields from a tumour segmention, that was taken
from the first Training data set of the BraTS 2020 data set. In that the brain area is identified with the integer value
of 1 and a possible mobile tumour cell concentration in the edema is marked with 2. Especially, the second computation, 
so the mapping of fields into the generated mesh, is quite intense and you might want to play with the 
volume_resolution. Furthermore, some structural elements of OncoFEM are shown, that help to investigate into whole 
studies.
"""
import oncofem as of
import os
########################################################################################################################
# INPUT
#
# In a first step an input needs to be defined. To do so, first a study object is created. This study then creates a
# workspace on the hard drive with two sub-folders 'der' and 'sol'. Herein, all derived  pre-processed results and final
# solutions are saved. The parent studies folder need to be set in the config.ini file. To compare the results of
# different subjects in a next hierarchical level a 'subject' needs to be created. This subject than can have several
# states of measurements taken at different time points. By means of that the initial state is created and the taken
# measurement files can be defined. A measurement can be created by the path to the relative file or directory and its
# modality. This ensures, that the information is interpreted correctly.
#
study = of.structure.Study("tut_03")
subj_1 = study.create_subject("Subject_1")
state_1 = subj_1.create_state("init_state")
data_dir = os.getcwd() + os.sep + "data" + os.sep
measure_1 = state_1.create_measure(data_dir + "mask.nii.gz", "mask")
#
# Before a field can be mapped, of course first the domain is needed, where any field can be mapped on. A field map
# generator entity is created and its working directory is set to the der-directory of state_1. The volume_resolution is
# set and with 'nii2dolfin_mesh, the medical NIFTI image is transformed into a mesh that can be taken from FEniCS. In 
# order to set the initial conditions right, the geometric dimension need to be set. All data is saved into a problem
# entity.
#
fmap = of.FieldMapGenerator(state_1.der_dir)
fmap.volume_resolution = 10
p = of.Problem()
p.geom.mesh = fmap.nii2dolfin_mesh(state_1.measures["mask"])
p.geom.dim = p.geom.mesh.geometric_dimension()
#
# Again, in this tutorial the tumour will be approximated as mobile cancer cells that spread inside the tissue. In this
# tutorial the mobile cancer cells inside the edema are approximated with an interpolation from a given mask. The mask
# is everywhere 1 in the brain area and 2 in the tumour area. The white and gray matter can be identified with the
# structure segmentation package OncoSTR and are simply given in this tutorial. These nifti files can be interpolated 
# analogously. First, minimum and maximum values are set and a mask is created from the image. This image also gives the
# affine, in order to transform any field to the same space. With 'interpolate()' the distribution is generated with the
# min value at the outside of the area and a max area at the center of the distribution. This interpolation can then be
# mapped into a field and load into FEniCS with the 'read_mapped_xdmf()' command. Since, these processes are quite 
# numerically expensive, their output will always be written to a file in the set directory and load back into with the
# named command.
#
min_val = 1.0E-13  # max concentration
max_val = 9.828212E-1  # max concentration
ede_mask = fmap.image2mask(state_1.measures["mask"], 2)
fmap.set_affine(state_1.measures["mask"])
ede_distr = "edema_distr"
out = fmap.interpolate(ede_mask, ede_distr, min_value=min_val, max_value=max_val)
mapped_edema = fmap.map_field(out)
p.geom.edema_distr = fmap.read_mapped_xdmf(mapped_edema)
p.geom.wm_distr = fmap.read_mapped_xdmf(data_dir + "white_matter.xdmf")
p.geom.gm_distr = fmap.read_mapped_xdmf(data_dir + "gray_matter.xdmf")
p.geom.csf_distr = fmap.read_mapped_xdmf(data_dir + "csf.xdmf")
#
# The next code lines are identical to the first tutorial. Be aware of the chosen solver type. The example in the paper 
# runs with mumps on a 128 GB nvidia cpu. Maybe this setting will not run on your computer and you should try 'lu' or 
# 'gmres'.
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
p.param.fem.solver_type = "mumps"  # Try "mumps", "lu" or "gmres" if error in solution
p.param.fem.maxIter = 20
p.param.fem.rel = 1E-7
p.param.fem.abs = 1E-8
# ADDITIONALS
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
model = of.base_models.TwoPhaseModel()
model.set_param(p)
model.set_function_spaces()
p.param.init.uS_0S = [0.0, 0.0, 0.0]
p.param.init.p_0S = 0.0
p.param.init.nS_0S = 0.75
cFt_0S = p.geom.edema_distr
p.param.add.cFkappa_0S = [cFt_0S]
bio_model = of.process_models.VerhulstKinetic()
bio_model.set_input(model.ansatz_functions)
prod_list = bio_model.get_output()
model.set_process_models(prod_list)
bounds = [(100.0, 129.0), (115.0, 160.0), (-20.0, 10.0)]
b1 = of.utils.fem.BoundingBox(p.geom.mesh, bounds)
p.geom.domain, p.geom.facet_function = of.utils.fem.mark_facet(p.geom.mesh, [b1])
p.geom.dim = p.geom.mesh.geometric_dimension()
bc_u_0 = of.utils.io.df.DirichletBC(model.function_space.sub(0).sub(0), 0.0, p.geom.facet_function, 1)
bc_u_1 = of.utils.io.df.DirichletBC(model.function_space.sub(0).sub(1), 0.0, p.geom.facet_function, 1)
bc_u_2 = of.utils.io.df.DirichletBC(model.function_space.sub(0).sub(2), 0.0, p.geom.facet_function, 1)
bc_p_0 = of.utils.io.df.DirichletBC(model.function_space.sub(1), 0.0, p.geom.facet_function, 1)
model.set_boundaries([bc_u_0, bc_u_1, bc_u_2, bc_p_0], None)
#
# SET AND SOLVE
#
model.set_structural_parameters()
model.set_weak_form()
of.utils.io.df.set_log_level(30)
model.set_solver()
model.set_initial_conditions(p.param.init, p.param.add)
model.solve()
