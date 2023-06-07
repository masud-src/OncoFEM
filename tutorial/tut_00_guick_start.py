"""
OncoFEM is a software 

Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""
import oncofem as of
import dolfin
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
# Set up tumour mapping
fmap.edema_min_value = 1.0  # max concentration
fmap.edema_max_value = 2.0  # max concentration
fmap.active_min_value = 1.0
fmap.active_max_value = 2.0
fmap.necrotic_min_value = 1.0
fmap.necrotic_max_value = 2.0
fmap.interpolation_method = "linear"  # nearest, cubic
#fmap.run_solid_tumor_mapping()
#fmap.run_edema_mapping()
fmap.mapped_act_file = fmap.fmap_dir + "active.xdmf"
fmap.mapped_nec_file = fmap.fmap_dir + "necrotic.xdmf"
fmap.mapped_ede_file =  fmap.fmap_dir + "edema.xdmf"
fmap.wms_mapping_method = "mean_averaged_value"  # "const_wm"
base_path = "/media/marlon/data/studies/tut_00/der/Subject_1/2023-05-23/white_matter_segmentation/tumor_class_pve_"
input = [base_path + "0.nii.gz", base_path + "1.nii.gz", base_path + "2.nii.gz"]
fmap.set_mixed_masks(input)
fmap.run_wm_mapping()
"""

"""
b1 = of.helper.BoundingBox(fmap.dolfin_mesh, (100.0, 129.0), (115.0, 160.0), (-20.0, 10.0))
p.geom.domain, p.geom.facet_function = fmap.mark_facet([b1])
p.geom.mesh = fmap.dolfin_mesh
p.geom.dim = 3
p.geom.edema_distr = of.helper.io.read_mapped_xdmf(fmap.mapped_ede_file)
p.geom.active_distr = of.helper.io.read_mapped_xdmf(fmap.mapped_act_file)
p.geom.necrotic_distr = of.helper.io.read_mapped_xdmf(fmap.mapped_nec_file)
p.geom.wm_distr = of.helper.io.read_mapped_xdmf(fmap.mapped_wm_file)
p.geom.gm_distr = of.helper.io.read_mapped_xdmf(fmap.mapped_gm_file)
p.geom.csf_distr = of.helper.io.read_mapped_xdmf(fmap.mapped_csf_file)
DFn_wm = 1e-4
DFn_gm = 1e-3
DFn_csf = 1e-2
DFt_wm = 0.00005
DFt_gm = 0.00005
DFt_csf = 0.00005
DFa_wm = 1e-12
DFa_gm = 1e-11
DFa_csf = 1e-10
p.param.mat.DFn_distr = of.helper.set_av_params([DFn_wm, DFn_gm, DFn_csf], [p.geom.wm_distr, p.geom.gm_distr, p.geom.csf_distr], [1, 1, 1])
p.param.mat.DFt_distr = of.helper.set_av_params([DFt_wm, DFt_gm, DFt_csf], [p.geom.wm_distr, p.geom.gm_distr, p.geom.csf_distr], [1, 1, 1])
p.param.mat.DFa_distr = of.helper.set_av_params([DFa_wm, DFa_gm, DFa_csf], [p.geom.wm_distr, p.geom.gm_distr, p.geom.csf_distr], [1, 1, 1])

p.param.gen.flag_defSplit = True

# time parameters
p.param.time.T_end = 200.0  # *86400
p.param.time.output_interval = 24.0/24.0  # *86400
p.param.time.dt = 3.0/24.0  # *86400

# material parameters base model
p.param.mat.rhoShR = 1190.0
p.param.mat.rhoStR = 1190.0  # muss größer sein als Sh
p.param.mat.rhoSnR = 1190.0
p.param.mat.rhoFR = 1993.3
p.param.mat.gammaFR = 1.0
p.param.mat.molFt = 2.018E13
p.param.mat.R = 8.31446261815324
p.param.mat.Theta = 37.0

# spatial varying material parameters
p.param.mat.lambdaSh = 3312.0
p.param.mat.lambdaSt = 3312.0
p.param.mat.lambdaSn = 3312.0
p.param.mat.muSh = 662.0
p.param.mat.muSt = 662.0
p.param.mat.muSn = 662.0
p.param.mat.kF = 5E-13
p.param.mat.DFt = 1.5E-13 * 86400

# FEM Paramereters
p.param.fem.solver_type = "lu"
p.param.fem.maxIter = 20
p.param.fem.rel = 1E-7
p.param.fem.abs = 1E-8

################################################################################################################
# ADDITIONALS
# material parameters
molFn = 0.18
DFn = 6.6E-10 * 86400
p.param.add.prim_vars = ["cFn"]
p.param.add.ele_types = ["CG"]
p.param.add.ele_orders = [1] 
p.param.add.tensor_orders = [0]
p.param.add.molFkappa = [molFn]
p.param.add.DFkappa = [DFn]
################################################################################################################
print("Start calculation")
dolfin.set_log_level(30)
model = of.modelling.base_model.Glioblastoma()
file = of.io.set_output_file(study.sol_dir + p.param.gen.title + "/TPM")
p.param.gen.output_file = file
model.set_param(p)
model.set_function_spaces()

################################################################################################################
# initial conditions
p.param.init.uS_0S = [0.0, 0.0, 0.0]
p.param.init.p_0S = 0.0
p.param.init.nSh_0S = 0.4 
p.param.init.nSt_0S = p.geom.active_distr
p.param.init.nSn_0S = p.geom.necrotic_distr
p.param.init.cFt_0S = p.geom.edema_distr  # field #fmg.read_mapped_xdmf(init_cFt)
cFn_0S = 1.0
cFa_0S = 0.0
p.param.add.cFkappa_0S = [cFn_0S]

################################################################################################################
# Bio chemical set up
bio_model = of.modelling.bio_chem_models.SimpleModel()
bio_model.set_prim_vars(model.ansatz_functions)
bio_model.set_param(p)
bio_model.flag_proliferation = True
bio_model.flag_metabolism = True
bio_model.flag_necrosis = True
bio_model.nSt_thres_lin_ms = 5e-5
bio_model.fac_nSt_lin_ms = 1e-1
bio_model.nu_Sh_necrosis = 1e-15 * 86400
bio_model.nu_St_necrosis = 1E-15 * 86400
bio_model.nu_Ft_necrosis = 0.0 * 86400
bio_model.cFn_min_necrosis = 0.85
bio_model.nSt_max = 0.5
bio_model.cFt_max = 9.828212E-1
bio_model.cFn_min_growth = 0.35
bio_model.nu_In_basal = 8.64e-28
bio_model.nu_Ft_proliferation = 0.0864
bio_model.nu_St_proliferation = 0.35856e-3  # 0.35856
bio_model.f_proli = 8.64e-5
prod_list = bio_model.return_prod_terms()
model.set_bio_chem_models(prod_list)
################################################################################################################
# Boundary conditions
# u (x,y,z), p, nSh, nSt, nSn, cFt, cFn, cFa
bc_u_0 = dolfin.DirichletBC(model.function_space.sub(0).sub(0), 0.0, p.geom.facet_function, 1)
bc_u_1 = dolfin.DirichletBC(model.function_space.sub(0).sub(1), 0.0, p.geom.facet_function, 1)
bc_u_2 = dolfin.DirichletBC(model.function_space.sub(0).sub(2), 0.0, p.geom.facet_function, 1)
bc_p_0 = dolfin.DirichletBC(model.function_space.sub(1), 0.0, p.geom.facet_function, 1)
#bc_cFn_1 = dolfin.DirichletBC(model.function_space.sub(6), 1.0, p.geom.facet_function, 4)
################################################################################################################

model.set_boundaries([bc_u_0, bc_u_1, bc_u_2, bc_p_0], None)
model.set_heterogenities()
model.set_weak_form()
model.set_solver()
model.set_initial_conditions(p.param.init, p.param.add)
model.solve()
