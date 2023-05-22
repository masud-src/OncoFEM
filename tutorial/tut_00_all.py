"""
In this second tutorial, the segmentation of mri images is shown.

Hierbei zunächst an Suditsch brain gezeigt, wie Generalisierung funktioniert.
Tumor Segmentierung mit modal agnostic type.
White Matter segmentierung.

Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""
import oncofem as of
"""
This time, we create a study called "tut_02" and create the Subject 1, with an initial state of two measurements. Since 
these measurements are raw data in the dcm format, first we need to perform the generalisation step. You can look at the
original images with fsleyes or comparable software and you will see, that the measurements are skew and the intensities
are quite dark, especially in the upper areas near the skull. 
"""
study = of.Study("tut_00")
subj_1 = study.create_subject("Subject_1")
state_1 = subj_1.create_state("init_state")
measure_1 = state_1.create_measure("data/BraTS/BraTS20_Training_001/BraTS20_Training_001_t1.nii.gz", "t1")
measure_2 = state_1.create_measure("data/BraTS/BraTS20_Training_001/BraTS20_Training_001_t1ce.nii.gz", "t1ce")
measure_3 = state_1.create_measure("data/BraTS/BraTS20_Training_001/BraTS20_Training_001_t2.nii.gz", "t2")
measure_4 = state_1.create_measure("data/BraTS/BraTS20_Training_001/BraTS20_Training_001_flair.nii.gz", "flair")
measure_5 = state_1.create_measure("data/BraTS/BraTS20_Training_001/BraTS20_Training_001_seg.nii.gz", "seg")
"""

"""
mri = of.MRI(state=state_1)
mri.load_measures()
mri.set_affine()
"""
Run solitary segmentation
"""
mri.set_tumor_segmentation()
mri.tumor_segmentation.infer_param.output_path = measure_5.dir_act
mri.tumor_segmentation.set_compartment_masks()
mri.tumor_segmentation.save_compartment_masks()
"""
White matter segmentation
"""
mri.set_wm_segmentation()
structural_input_files = [mri.t1_dir]  # , mri_2.t1ce_dir, mri_2.t2_dir, mri_2.flair_dir]
mri.wm_segmentation.tumor_handling_approach = "mean_averaged_value"  # tumor_entity_weighted" #mean_averaged_value"
mri.wm_segmentation.set_input_wm_seg(structural_input_files)
mri.wm_segmentation.run() 
"""

"""
p = of.Problem(mri)
"""
Field mapping
"""
run_fmp = True
if run_fmp:
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
    fmap.run_solid_tumor_mapping()
    fmap.run_edema_mapping()
    fmap.wms_mapping_method = "mean_averaged_value"  # "const_wm"
    base_path = "/media/marlon/data/studies/tut_00/der/Subject_1/2023-05-22/white_matter_segmentation/tumor_class_pve_"
    input = [base_path + "0.nii.gz", base_path + "1.nii.gz", base_path + "2.nii.gz"]
    fmap.set_mixed_masks(input)
    fmap.run_wm_map()
"""

"""
b1 = of.modelling.field_map_generator.BoundingBox(fmap.dolfin_mesh, (100.0, 129.0), (115.0, 160.0), (-20.0, 10.0))
b2 = of.modelling.field_map_generator.BoundingBox(fmap.dolfin_mesh, (78.0, 95.0), (154.0, 165.0), (-20.0, 20.0))
p.geom.domain, p.geom.facet_function = fmap.mark_facet([b1, b2])
p.geom.mesh = fmap.dolfin_mesh
p.geom.dim = 3
#p.geom.edema_distr = fmg.read_mapped_xdmf(fmg.mapped_edema_file)
#p.geom.solid_tumor_distr = fmg.read_mapped_xdmf(fmg.mapped_solid_tumor_file)
#p.geom.necrotic_distr = fmg.read_mapped_xdmf(fmg.mapped_necrotic_file)