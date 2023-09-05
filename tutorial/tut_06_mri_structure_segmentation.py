"""

"""
import oncofem as of
########################################################################################################################
# INPUT
study = of.helper.structure.Study("tut_06")
subj = study.create_subject("BraTS_01")
state = subj.create_state("pre_operation_1")
path = of.ONCOFEM_DIR + "/data/tutorial/BraTS/BraTS20_Training_001/"
state.create_measure(path + "BraTS20_Training_001_t1.nii.gz", "t1")
state.create_measure(path + "BraTS20_Training_001_seg.nii.gz", "seg")

mri = of.mri.MRI(state)
mri.work_dir = study.der_dir

mri.set_tumor_segmentation()
mri.tumor_segmentation.seg_file = state.measures[1].dir_act
mri.tumor_segmentation.set_compartment_masks()
########################################################################################################################
# MRI STRUCTURE SEGMENTATION
mri.set_structure_segmentation()
mri.structure_segmentation.tumor_handling_approach = "bias_corrected" 
structural_input_files = [mri.t1_dir]
mri.structure_segmentation.set_input_structure_seg(structural_input_files)
mri.structure_segmentation.run()
