"""
MRI structure segmentation tutorial

In order to perform the structure segmentation first a study is set in order to generate a workspace and an input state 
is initialized that can be handed to the mri object. Before the structure segmentation can be used first the actual
segmentation of the tumor distribution needs to be set via initializing of the tumor segmentation entity. Here, the
segmentation file argument can be set manually and to identify the different zones of the tumor, the compartment masks 
are evaluated using 'set_compartment_masks()'.
Again, first the object needs to be initialized and bind to its structural entity. To do so, the command
'set_structure_segmentation' is used and the method about regarding the tumor distributed area can be chosen. Here, the
user has the choice between the 'tumor_entity_weighted' approach from tutorial "tut_01_quickstart" or use the 
"bias_corrected" approach. For input of the structure segmentation the user can change the list of the structural input
files. Best results have been with just the t1 modality.

Besides the choice of the handling of the tumor distributed area, the user can set the number of the healthy and tumor 
tissue classes with:

mri.structure_segmentation.tumor_handling_classes = 3
mri.structure_segmentation.brain_handling_classes = 3 

The default is set to both to 3.
"""
import oncofem as of
########################################################################################################################
# INPUT
study = of.helper.structure.Study("tut_06")
subj = study.create_subject("BraTS_01")
state = subj.create_state("pre_operation_1")
path = of.ONCOFEM_DIR + "/data/tutorial/BraTS/BraTS20_Training_001/"
state.create_measure(path + "BraTS20_Training_001_t1.nii.gz", "t1")
state.create_measure(path + "BraTS20_Training_001_t1ce.nii.gz", "t1ce")
state.create_measure(path + "BraTS20_Training_001_t2.nii.gz", "t2")
state.create_measure(path + "BraTS20_Training_001_flair.nii.gz", "flair")
state.create_measure(path + "BraTS20_Training_001_seg.nii.gz", "seg")
########################################################################################################################
# TUMOR SEGMENTATION
mri = of.mri.MRI(state)
mri.work_dir = study.der_dir
mri.set_tumor_segmentation()
mri.tumor_segmentation.seg_file = state.measures[4].dir_act
mri.tumor_segmentation.set_compartment_masks()
########################################################################################################################
# STRUCTURE SEGMENTATION
mri.set_structure_segmentation()
mri.structure_segmentation.tumor_handling_approach = "bias_corrected" 
structural_input_files = [mri.t1_dir]
mri.structure_segmentation.set_input_structure_seg(structural_input_files)
mri.structure_segmentation.run()
