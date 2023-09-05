"""

"""
import oncofem as of
from oncofem.helper.fem_aux import BoundingBox

study = of.Study("tut_03")
subj = study.create_subject("Subject_1")
state = subj.create_state("init_state")
measure_1 = state.create_measure("tutorial/BraTS/BraTS20_Training_001/BraTS20_Training_001_t1.nii.gz", "t1")
measure_2 = state.create_measure("tutorial/BraTS/BraTS20_Training_001/BraTS20_Training_001_t1ce.nii.gz", "t1ce")
measure_3 = state.create_measure("tutorial/BraTS/BraTS20_Training_001/BraTS20_Training_001_t2.nii.gz", "t2")
measure_4 = state.create_measure("tutorial/BraTS/BraTS20_Training_001/BraTS20_Training_001_flair.nii.gz", "flair")
measure_5 = state.create_measure("tutorial/BraTS/BraTS20_Training_001/BraTS20_Training_001_seg.nii.gz", "seg")

mri = of.mri.MRI(state)
mri.load_measures()
mri.set_affine()
mri.tumor_segmentation = of.mri.mri.TumorSegmentation(mri)
mri.tumor_segmentation.infer_param.output_path = "/tutorial/tutorial/BraTS/BraTS20_Training_001/BraTS20_Training_001_seg.nii.gz"
mri.tumor_segmentation.set_compartment_masks()
"""
copy images, because already generalised
"""
run_cp = False
if run_cp:
    print("cp " + measure_1.dir_act + " " + mri.generalisation.dir + ".")
    of.helper.run_shell_command("cp " + measure_1.dir_act + " " + mri.generalisation.dir + ".")


run_wms = True
if run_wms:#
    structural_input_files = [mri.t1_dir]#, mri_2.t1ce_dir, mri_2.t2_dir, mri_2.flair_dir]
    mri.set_wm_segmentation()
    mri.structure_segmentation.tumor_handling_approach = "tumor_entity_weighted" #mean_averaged_value"
    mri.structure_segmentation.set_input_structure_seg(structural_input_files)
    mri.structure_segmentation.run() 

p = of.Problem(mri)

# Field mapping
run_fmp = True
if run_fmp:
    fmap = of.modelling.FieldMapGenerator(p)
    # Set up geometry
    fmap.volume_resolution = 80
    fmap.generate_geometry_file(p.mri.t1_dir)
    b1 = BoundingBox(fmap.dolfin_mesh, (100.0, 129.0), (115.0, 160.0), (-20.0, 10.0))
    p.geom.domain, p.geom.facet_function = fmap.mark_facet([b1])
    p.geom.mesh = fmap.dolfin_mesh
    p.geom.dim = 3
    # Set up tumour mapping
    fmap.edema_min_value = 1.0 
    fmap.edema_max_value = 2.0 
    fmap.active_min_value = 1.0
    fmap.active_max_value = 2.0
    fmap.necrotic_min_value = 1.0
    fmap.necrotic_max_value = 2.0
    fmap.interpolation_method = "linear"  # nearest
    fmap.run_solid_tumor_mapping()
    fmap.run_edema_mapping()
    #fmap.set_mixed_masks()
    #fmap.run_wm_mapping()
    #p.geom.edema_distr = fmg.read_mapped_xdmf(fmg.mapped_edema_file)
    #p.geom.solid_tumor_distr = fmg.read_mapped_xdmf(fmg.mapped_solid_tumor_file)
    #p.geom.necrotic_distr = fmg.read_mapped_xdmf(fmg.mapped_necrotic_file)
