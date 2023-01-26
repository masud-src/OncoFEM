# Imports
import os, time
from oncofem.helper.general import set_working_folder
from oncofem.struct.study import Study
from oncofem.struct.problem import Problem
from oncofem.mri.white_matter_segmentation import WhiteMatterSegmentation
from oncofem.modelling.field_map_generator.field_map_generator import FieldMapGenerator
from oncofem.helper import io
from oncofem.modelling.base_model.continuum_mechanics.tpm.glioblastoma import Glioblastoma
import dolfin as df
import ufl


#Define Study
study = Study("milestone")

# Defining of general Problem
x = Problem()

folder = "/media/marlon/data/MRI_data/UPENN-GBM-00001_11/UPENN-GBM-00001_11_"
x.mri.t1_dir          = folder + "T1.nii.gz"
x.mri.t1ce_dir        = folder + "T1GD.nii.gz"
x.mri.t2_dir          = folder + "T2.nii.gz"
x.mri.flair_dir       = folder + "FLAIR.nii.gz"
x.mri.tumor_seg_dir   = folder + "automated_approx_segm.nii.gz"
#x.mri.dti_ad_dir      = folder + "DTI.nii.gz"
#x.mri.dti_fa_dir      = folder + "DTI_FA.nii.gz"
#x.mri.dti_rd_dir      = folder + "DTI_RD.nii.gz"
#x.mri.dti_tr_dir      = folder + "DTI_TR.nii.gz"
#x.mri.dsc_psr_dir     = folder + "DSC_PSR.nii.gz"
#x.mri.dsc_ph_dir      = folder + "DSC_PH.nii.gz"
#x.mri.dsc_ap_dir      = folder + "DSC_ap-rCBV.nii.gz"

# for subject
working_folder = set_working_folder(study.der_dir + "W1" + os.sep)

# White matter segmentation
run_wms = False
if run_wms:
    working_folder = set_working_folder(working_folder + "wms" + os.sep)
    structural_input_files = [x.mri.t1_dir, x.mri.t2_dir]
    wms = WhiteMatterSegmentation(study)
    wms.set_input_wm_seg(structural_input_files, x.mri.tumor_seg_dir, work_dir=working_folder)
    wms.run_wm_seg(x) 

# Field mapping
run_fmp = True
if run_fmp:
    subject_dir = study.der_dir + "W1" + os.sep
    fmg = FieldMapGenerator(study)
    # Set up geometry
    fmg.set_general(t1_dir=x.mri.t1_dir, work_dir=subject_dir)
    fmg.volume_resolution = 8#20
    fmg.generate_geometry_file()
    x.geom.domain, x.geom.facet_function = fmg.set_fixed_boundary(x_bounds=(106.0, 129.0), y_bounds=(130, 148), z_bounds=(-2, 6))
    # Set up tumour mapping
    fmg.tumor_seg_file = x.mri.tumor_seg_dir
    tmg = fmg.set_up_tumor_map_generator()
    tmg.max_edema_value = 1.0E-4  # max concentration
    tmg.max_solid_tumor_value = 0.4  # max solid tumor
    tmg.max_necrotic_value = 0.5  # max necrotic core
    fmg.generate_tumor_map()
    x.geom.edema_distr = fmg.read_mapped_xdmf(fmg.mapped_edema_file)
    x.geom.solid_tumor_distr = fmg.read_mapped_xdmf(fmg.mapped_solid_tumor_file)
    x.geom.necrotic_distr = fmg.read_mapped_xdmf(fmg.mapped_necrotic_file)
    # Set up white matter mapping
    fmg.wms_dir = working_folder + "wms" + os.sep
    fmg.wms_mapping_handler = 0
    fmg.generate_wms_map()
    x.geom.wm_distr = fmg.read_mapped_xdmf(fmg.mapped_wm_file)
    x.geom.gm_distr = fmg.read_mapped_xdmf(fmg.mapped_gm_file)
    x.geom.csf_distr = fmg.read_mapped_xdmf(fmg.mapped_csf_file)
    DFn_wm = 1e-4
    DFn_gm = 1e-3
    DFn_csf = 1e-2
    DFt_wm = 0.00005
    DFt_gm = 0.00005
    DFt_csf = 0.00005
    DFa_wm = 1e-12
    DFa_gm = 1e-11
    DFa_csf = 1e-10
    x.param.mat.DFn_distr = fmg.set_av_params([DFn_wm, DFn_gm, DFn_csf], [x.geom.wm_distr, x.geom.gm_distr, x.geom.csf_distr], [1, 1, 1])
    x.param.mat.DFt_distr = fmg.set_av_params([DFt_wm, DFt_gm, DFt_csf], [x.geom.wm_distr, x.geom.gm_distr, x.geom.csf_distr], [1, 1, 1])
    x.param.mat.DFa_distr = fmg.set_av_params([DFa_wm, DFa_gm, DFa_csf], [x.geom.wm_distr, x.geom.gm_distr, x.geom.csf_distr], [1, 1, 1])
    #fieldmapper.generate_dti_map()
    #fieldmapper.generate_dsc_map()

x.geom.mesh = x.geom.facet_function.mesh()
x.geom.dx = df.Measure("dx", metadata={'quadrature_degree': 2})

print("Num vertices: ", x.geom.facet_function.mesh().num_vertices())
print("Num cells: ", x.geom.facet_function.mesh().num_cells())

# Parameter settings
# General infos
x.param.gen.flag_proliferation = True
x.param.gen.flag_metabolism = False
x.param.gen.flag_apop = False
x.param.gen.flag_necrosis = False
x.param.gen.flag_defSplit = False

# Material Parameters
x.param.mat.nS_0S = 0.4
x.param.mat.nSt_0S = 0.0
x.param.mat.rhoShR = 500
x.param.mat.rhoStR = 950
x.param.mat.rhoSnR = 1000
x.param.mat.rhoFR = 1000
x.param.mat.gammaFR = 1000
x.param.mat.molFn = 1
x.param.mat.molFt = 1
x.param.mat.molFa = 1
x.param.mat.kF = 1e-3
x.param.mat.lambdaSh = 1e7
x.param.mat.lambdaSt = 1.5e7
x.param.mat.lambdaSn = 1e7
x.param.mat.muSh = 1e7
x.param.mat.muSt = 1e7
x.param.mat.muSn = 1e7

# Time Parameters
x.param.time.T_end = 3600*24*30
x.param.time.dt = 3600*24

# FEM Paramereters
x.param.fem.solver_param.newton.solver_type = "mumps"
x.param.fem.solver_param.newton.maxIter = 10
x.param.fem.solver_param.newton.rel = 1E-7
x.param.fem.solver_param.newton.abs = 1E-8
x.param.fem.type_u = "CG"
x.param.fem.order_u = 1
x.param.fem.type_p = "CG"
x.param.fem.order_p = 1
x.param.fem.type_nSh = "CG"
x.param.fem.order_nSh = 1
x.param.fem.type_nSt = "CG"
x.param.fem.order_nSt = 1
x.param.fem.type_nSn = "CG"
x.param.fem.order_nSn = 1
x.param.fem.type_cFn = "CG"
x.param.fem.order_cFn = 1
x.param.fem.type_cFt = "CG"
x.param.fem.order_cFt = 1
x.param.fem.type_cFa = "CG"
x.param.fem.order_cFa = 1
x.param.fem.order_I1 = 1
x.param.fem.order_I2 = 1
x.param.fem.order_I3 = 1

print("Start calculation")
df.set_log_level(30)

model = Glioblastoma()
x.param.gen.title = "W1"
file = io.set_output_file(study.sol_dir + x.param.gen.title + "/TPM")
x.param.gen.output_file = file
model.set_param(x)
model.set_function_spaces()

u, p, nSh, nSt, nSn, cFn, cFt, cFa = df.split(model.ansatz_functions)

def linear(field, alpha):
    return field * alpha

def delta2max(field, max, dt):
    return max * dt - field

def verhulst_growth(field, kappa, max_value):
    """
    #solves 
    #field * kappa * (1 - field / max_value)
    """
    return field * kappa * (1 - field / max_value)

tres_cFn_survival = df.Constant(0.0008)
tres_cFn_necrosis = df.Constant(0.05)
tres_cFt_bulk = df.Constant(0.006)
alpha_metabolism_cFt = df.Constant(0.5e-5)
alpha_metabolism_nSt = df.Constant(1e-2)
# Define conditions
low_cFn_survival = ufl.le(cFn, tres_cFn_survival)
high_cFn_survival = ufl.gt(cFn, tres_cFn_survival)
low_cFn_necrosis = ufl.le(cFn, tres_cFn_necrosis)
mobile_cells_larger_zero = ufl.gt(cFt, df.DOLFIN_EPS)
tumor_larger_zero = ufl.gt(cFt, df.DOLFIN_EPS)  # tumor_larger_zero = ufl.gt(cFt, df.DOLFIN_EPS)
cFt_is_bulk = ufl.gt(cFt, tres_cFt_bulk)

x.bmm.bm_model_prolif_cFt = ufl.conditional(high_cFn_survival, verhulst_growth(cFt, 1e-2, 0.008), 0.0)  # TODO condition upside down
x.bmm.bm_model_prolif_nSt = ufl.conditional(high_cFn_survival, ufl.conditional(cFt_is_bulk, 2.0, 0.0), 0.0)  # TODO condition upside down
x.bmm.bm_model_necros_nSh = - ufl.conditional(low_cFn_necrosis, 1.0, 0.0)
x.bmm.bm_model_necros_nSt = 0.0#-ufl.conditional(low_cFn_necrosis, 3.0, 0.0)
x.bmm.bm_model_necros_cFt = -ufl.conditional(low_cFn_necrosis, x.bmm.bm_model_prolif_nSt, 0.0)
x.bmm.bm_model_apopto_nSh = 0.0
x.bmm.bm_model_apopto_nSt = 0.0
x.bmm.bm_model_apopto_cFt = 0.0
x.bmm.bm_model_apopto_cFa = 0.0
bm_model_metabo_cFt = linear(cFt, alpha_metabolism_cFt)
bm_model_metabo_nSt = linear(nSt, alpha_metabolism_nSt)
x.bmm.bm_model_metabo_cFn = - bm_model_metabo_cFt - bm_model_metabo_nSt
model.set_bio_chem_models(x)
########################################################
# Boundary conditions
# u (x,y,z), p, nSh, nSt, nSn, cIn, cIt, cIv, cIa
bc_u_0 = df.DirichletBC(model.function_space.sub(0).sub(0), 0.0, x.geom.facet_function, 1)
bc_u_1 = df.DirichletBC(model.function_space.sub(0).sub(1), 0.0, x.geom.facet_function, 1)
bc_u_2 = df.DirichletBC(model.function_space.sub(0).sub(2), 0.0, x.geom.facet_function, 1)
bc_cFn_1 = df.DirichletBC(model.function_space.sub(5), 1e-1, x.geom.facet_function, 1)
bc_cFn_2 = df.DirichletBC(model.function_space.sub(5), 1e-1, x.geom.facet_function, 0)
model.set_boundaries([bc_u_0, bc_u_1, bc_u_2, bc_cFn_1], None)
model.set_initial_condition()
start = time.time()  # start time
model.solve()
end = time.time()  # start time
print(end-start)
