"""

"""
import oncofem as of
import os

study = of.helper.structure.Study("tut_5")

mri_2.tumor_segmentation.infer_param.output_path = "/home/marlon/Software/OncoFEM/tutorial/data/BraTS/BraTS20_Training_001/BraTS20_Training_001_seg.nii.gz"
mri_2.tumor_segmentation.set_compartment_masks()
"""

"""
path = state_2.study_dir + of.helper.DER_DIR + state_2.subject + os.sep + state_2.dir + "wms" + os.sep
working_folder = of.helper.general.mkdir_if_not_exist(path)
structural_input_files = [mri_2.t1_dir]
mri_2.set_wm_segmentation()
mri_2.structure_segmentation.tumor_handling_approach = "bias_corrected"
mri_2.structure_segmentation.set_input_structure_seg(structural_input_files)
mri_2.structure_segmentation.run() 
