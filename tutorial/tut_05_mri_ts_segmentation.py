"""
MRI tumor segmentation inference tutorial

To initialize the inference of the tumor segmentation just a study is set, in order to generate a workspace. And an 
input state is initialized. Here the BraTS_001 dataset is chosen, which can be used for assessing the segmentation 
quality, see our respective paper. Since, the tumor segmentation is part of the mri module, such an object needs to be 
initialized and the directory is manually set.

The tumor segmentation module is initialized via a setter method. The inference takes the images that are hold by the 
mri object. With the initialization of the inference, the best model is automatically chosen, depending on the given 
modalities. The preferred one of course is the full modality mode, followed by the t1gd and t1 models. Of course, the 
used model can be changed because of the users choice. Therefore, the config attribute of the segmentation can be 
manually set. For test purposes the user can chose different models with a switch 'select_model'. Note, that each model
was trained on a gpu (Nvidia a40, 48 GB VRAM, 32 core AMD epyc type 7452) and only runs with comparable hardware.
"""
import oncofem as of
########################################################################################################################
# INPUT
study = of.helper.structure.Study("tut_05")
subj = study.create_subject("BraTS_01")
state = subj.create_state("pre_operation_1")
path = "/tutorial/tutorial/BraTS/BraTS20_Training_001/"
state.create_measure(path + "BraTS20_Training_001_t1.nii.gz", "t1")
state.create_measure(path + "BraTS20_Training_001_t1ce.nii.gz", "t1ce")
state.create_measure(path + "BraTS20_Training_001_t2.nii.gz", "t2")
state.create_measure(path + "BraTS20_Training_001_flair.nii.gz", "flair")
########################################################################################################################
# TUMOR SEGMENTATION
mri = of.mri.MRI(state)
mri.work_dir = study.der_dir
mri.set_tumor_segmentation()
mri.tumor_segmentation.init_inference()
select_model = False
if select_model:
    model = "full"  # "full", "t1", "t1gd", "t2", "flair"
    mri.tumor_segmentation.config = of.ONCOFEM_DIR + "/data/tumor_segmentation/" + model + "/hyperparam.yaml"
mri.tumor_segmentation.run_inference()
