"""

"""
import oncofem as of


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
mri.set_tumor_segmentation()

mri.tumor_segmentation.input_data = "/home/marlon/Software/OncoFEM/tutorial/tutorial/BraTS/"
mri.tumor_segmentation.init_inference()
#path = "/media/marlon/tutorial/studies/tut_04/der/tumor_segmentation/full_neural_net/"
#mri.tumor_segmentation.config = path + "hyperparam.yaml"
mri.tumor_segmentation.run_inference()
