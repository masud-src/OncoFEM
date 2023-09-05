"""
MRI tumor segmentation inference tutorial

To initialize the training of the tumor segmentation just a study is set, in order to generate a workspace. An input 
state is initialized. Here the BraTS_001 dataset is chosen, which can be used for assessing the segmentation quality.
Since, the tumor segmentation is part of the mri module, such an object needs to be initialized and the directory is 
manually set.

The tumor segmentation module is initialized via a setter method. The inference takes the images that are hold by the 
mri object. With the initialization of the inference, the best model is automatically chosen, depending on the given 
modalities. The preferred one of course is the full modality mode, followed by the t1gd and t1 models. Of course, the 
used model can be changed because of the users choice. Therefore, the config attribute of the segmentation can be 
manually set. For test purposes the commented out lines can be inserted again. By swapping the paths, a different model 
is selected in each case.
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

mri = of.mri.MRI(state)
mri.work_dir = study.der_dir
########################################################################################################################
# TUMOR SEGMENTATION
mri.set_tumor_segmentation()
mri.tumor_segmentation.init_inference()
#path = "/media/marlon/tutorial/studies/tut_04/der/tumor_segmentation/t1/"
#path = "/media/marlon/tutorial/studies/tut_04/der/tumor_segmentation/t1gd/"
#path = "/media/marlon/tutorial/studies/tut_04/der/tumor_segmentation/t2/"
#path = "/media/marlon/tutorial/studies/tut_04/der/tumor_segmentation/flair/"
#mri.tumor_segmentation.config = path + "hyperparam.yaml"
mri.tumor_segmentation.run_inference()
