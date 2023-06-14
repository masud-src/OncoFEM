"""
OncoFEM is a software 

Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""
import os
import oncofem as of
import dolfin as df

import oncofem.helper.io

"""
Definition of Input 

Initiate study and a subject with an input state. The input state holds 4 measures
"""
study = of.Study("tut_00")
subj_1 = study.create_subject("Subject_1")
state_1 = subj_1.create_state("init_state")
measure_1 = state_1.create_measure("data/BraTS/BraTS20_Training_001/BraTS20_Training_001_t1.nii.gz", "t1")
measure_2 = state_1.create_measure("data/BraTS/BraTS20_Training_001/BraTS20_Training_001_seg.nii.gz", "seg")
"""
From that the mri entitity of oncofem can be set up

First is the initialisation with input state

Measures need to be loaded in sequence t1, t1ce, t2, flair

affine is set based on first loaded measure

since already generalised and segmented nothing more is needed
"""
mri = of.MRI(state=state_1)
mri.load_measures()
mri.set_affine()
"""
Tumor segmentation
"""
mri.set_tumor_segmentation()
mri.tumor_segmentation.infer_param.output_path = measure_2.dir_act
mri.tumor_segmentation.set_compartment_masks()
"""
White matter segmentation
"""
# Set up white matter segmentation
run_wms = False
if run_wms:
    working_folder = of.helper.mkdir_if_not_exist(study.dir + of.helper.DER_DIR + subj_1.ident + os.sep + state_1.dir + "wms" + os.sep)
    structural_input_files = [mri.t1_dir]  # , mri_2.t1ce_dir, mri_2.t2_dir, mri_2.flair_dir]
    mri.set_wm_segmentation()
    mri.wm_segmentation.tumor_handling_approach = "tumor_entity_weighted"  # mean_averaged_value"
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
"""
Set up problem with hand over mri state, not necessary
"""
p = of.Problem(mri)
p.param.gen.title = "Subject_1"
"""
Field mapping
"""
fmap = of.modelling.FieldMapGenerator(p)
# Set up geometry
fmap.volume_resolution = 2#20
fmap.generate_geometry_file(p.mri.t1_dir)
# Set up tumor mapping
run_tumor_mapping = True
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
"""

"""
b1 = of.helper.BoundingBox(fmap.dolfin_mesh, (100.0, 129.0), (115.0, 160.0), (-20.0, 10.0))
p.geom.domain, p.geom.facet_function = fmap.mark_facet([b1])
p.geom.mesh = fmap.dolfin_mesh
p.geom.dim = 3
p.geom.edema_distr = of.helper.io.read_mapped_xdmf(fmap.mapped_ede_file)
p.geom.wm_distr = of.helper.io.read_mapped_xdmf(fmap.mapped_wm_file)
p.geom.gm_distr = of.helper.io.read_mapped_xdmf(fmap.mapped_gm_file)
p.geom.csf_distr = of.helper.io.read_mapped_xdmf(fmap.mapped_csf_file)

################################################################################################################
# BASE MODEL
# general info
p.param.gen.flag_defSplit = True

# time parameters
p.param.time.T_end = 29.0  # *86400
p.param.time.output_interval = 24.0/24.0  # *86400
p.param.time.dt = 3.0/24.0  # *86400

# material parameters base model
p.param.mat.rhoSR = 1190.0
p.param.mat.rhoFR = 1993.3
p.param.mat.gammaFR = 1.0
p.param.mat.R = 8.31446261815324
p.param.mat.Theta = 37.0

# spatial varying material parameters
p.param.mat.lambdaS = 3312.0
p.param.mat.muS = 662.0
p.param.mat.kF = 5E-13

# FEM Paramereters
p.param.fem.solver_type = "lu"
p.param.fem.maxIter = 20
p.param.fem.rel = 1E-7
p.param.fem.abs = 1E-8
################################################################################################################

################################################################################################################
# ADDITIONALS
# material parameters
molFt = 2.018E13
DFt_wm = 1e-12
DFt_gm = 1e-11
DFt_csf = 1e-10
DFt = of.helper.set_av_params([DFt_wm, DFt_gm, DFt_csf], [p.geom.wm_distr, p.geom.gm_distr, p.geom.csf_distr], [1, 1, 1])
p.param.add.prim_vars = ["cFt"]
p.param.add.ele_types = ["CG"]
p.param.add.ele_orders = [1] 
p.param.add.tensor_orders = [0]
p.param.add.molFkappa = [molFt]
p.param.add.DFkappa = [DFt]
################################################################################################################
print("Start calculation")
df.set_log_level(30)
model = oncofem.modelling.base_model.TwoPhaseModel()
file = oncofem.helper.io.set_output_file(study.sol_dir + p.param.gen.title + "/TPM")
p.param.gen.output_file = file
model.set_param(p)
model.set_function_spaces()

################################################################################################################
# initial conditions
p.param.init.uS_0S = [0.0, 0.0, 0.0]
p.param.init.p_0S = 0.0
p.param.init.nS_0S = 0.4 
cFt_0S = p.geom.edema_distr
p.param.add.cFkappa_0S = [cFt_0S]

################################################################################################################
# Bio chemical set up
bio_model = oncofem.modelling.bio_chem_models.GompertzKinetic()
bio_model.set_prim_vars(model.ansatz_functions)
bio_model.max_cFt = 9.828212E-1
bio_model.init_f = 1.
bio_model.speed = 1.
prod_list = bio_model.return_prod_terms()
model.set_bio_chem_models(prod_list)
################################################################################################################
# Boundary conditions
# u (x,y,z), p, nSh, nSt, nSn, cFt, cFn, cFa
bc_u_0 = df.DirichletBC(model.function_space.sub(0).sub(0), 0.0, p.geom.facet_function, 1)
bc_u_1 = df.DirichletBC(model.function_space.sub(0).sub(1), 0.0, p.geom.facet_function, 1)
bc_u_2 = df.DirichletBC(model.function_space.sub(0).sub(2), 0.0, p.geom.facet_function, 1)
bc_p_0 = df.DirichletBC(model.function_space.sub(1), 0.0, p.geom.facet_function, 1)
################################################################################################################

model.set_boundaries([bc_u_0, bc_u_1, bc_u_2, bc_p_0], None)
model.set_heterogenities()
model.set_weak_form()
model.set_solver()
model.set_initial_conditions(p.param.init, p.param.add)
model.solve() 
