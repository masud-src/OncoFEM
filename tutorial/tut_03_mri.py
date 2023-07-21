"""
In this second tutorial, the segmentation of mri images is shown.

Hierbei zunächst an Suditsch brain gezeigt, wie Generalisierung funktioniert.
Tumor Segmentierung mit modal agnostic type.
White Matter segmentierung.

This time, we create a study called "tut_02" and create the Subject 1, with an initial state of two measurements. Since 
these measurements are raw data in the dcm format, first we need to perform the generalisation step. You can look at the
original images with fsleyes or comparable software and you will see, that the measurements are skew and the intensities
are quite dark, especially in the upper areas near the skull. 

co-registration performs resampling automatically

tumour segmentation

copy images, because already generalised

Train neural net

Train again with reduced input data

Train again with reduced input data and randomised blank ref image

Run solitary segmentation

White matter segmentation

Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""
import oncofem as of
import datetime
import os

study = of.struc.Study("tut_02")
"""
subj_1 = study.create_subject("Subject_1")
state_1 = subj_1.create_state("init_state", datetime.date.today())
measure_1 = state_1.create_measure("data/Suditsch/T1", "t1")
measure_2 = state_1.create_measure("data/Suditsch/Flair", "flair")
"""

"""
mri = of.MRI(state_1)
"""

"""
mri.set_generalisation()
for measure in [measure_1, measure_2]:
    mri.generalisation.dcm2niigz(measure)
    mri.generalisation.bias_correction(measure)
"""

"""
print("begin coregister")
mri.generalisation.coregister_modality2atlas()   
print("begin skull strip")
mri.generalisation.skull_strip()
"""

"""
mri.t1_dir = measure_1.dir_act
mri.flair_dir = measure_2.dir_act
"""

"""
mri.generalisation.run_all()


"""
subj_2 = study.create_subject("Subject_2")
state_2 = subj_2.create_state("init_state")
measure_21 = state_2.create_measure("data/BraTS/BraTS20_Training_001/BraTS20_Training_001_t1.nii.gz", "t1")
measure_22 = state_2.create_measure("data/BraTS/BraTS20_Training_001/BraTS20_Training_001_t1ce.nii.gz", "t1ce")
measure_23 = state_2.create_measure("data/BraTS/BraTS20_Training_001/BraTS20_Training_001_t2.nii.gz", "t2")
measure_24 = state_2.create_measure("data/BraTS/BraTS20_Training_001/BraTS20_Training_001_flair.nii.gz", "flair")
measure_25 = state_2.create_measure("data/BraTS/BraTS20_Training_001/BraTS20_Training_001_seg.nii.gz", "seg")
"""

"""
mri_2 = of.mri.MRI(state=state_2)
mri_2.load_measures()
mri_2.set_affine()
mri_2.set_tumor_segmentation()
"""

"""
run_cp = False
if run_cp:
    print("cp " + measure_21.dir_act + " " + mri_2.generalisation.dir + ".")
    of.helper.general.run_shell_command("cp " + measure_21.dir_act + " " + mri_2.generalisation.dir + ".")
"""

"""
run_train = False
if run_train:
    mri_2.tumor_segmentation.train_param.save_folder = "full_neural_net"
    mri_2.tumor_segmentation.train_param.data_folder = "/home/marlon/Software/OncoFEM/tutorial/data/BraTS"
    mri_2.tumor_segmentation.train_param.input_patterns = ["_t1", "_t1ce", "_t2", "_flair"]
    mri_2.tumor_segmentation.run_training()
"""

"""
run_train2 = False
if run_train2:
    mri_2.tumor_segmentation.train_param.save_folder = "t1_t2_fl_neural_net"
    mri_2.tumor_segmentation.train_param.data_folder = "/home/marlon/Software/OncoFEM/tutorial/data/BraTS"
    mri_2.tumor_segmentation.train_param.input_patterns = ["_t1", "_t2", "_flair"]
    mri_2.tumor_segmentation.run_training()
"""

"""
run_train3 = False
if run_train3:
    mri_2.tumor_segmentation.train_param.save_folder = "t1_t2_fl_neural_net"
    mri_2.tumor_segmentation.train_param.rand_blank = True
    mri_2.tumor_segmentation.train_param.data_folder = "/home/marlon/Software/OncoFEM/tutorial/data/BraTS"
    mri_2.tumor_segmentation.train_param.input_patterns = ["_t1", "_t1ce", "_t2", "_flair"]
    mri_2.tumor_segmentation.run_training()
"""

"""
run_seg = False
if run_seg:
    mri_2.tumor_segmentation.infer_param.input_patterns = ["_t1", "_t1ce", "_t2", "_flair"]
    mri_2.tumor_segmentation.infer_param.input_data = mri_2.generalisation.dir
    mri_2.tumor_segmentation.run_segmentation()
else:
    mri_2.tumor_segmentation.infer_param.output_path = "/home/marlon/Software/OncoFEM/tutorial/data/BraTS/BraTS20_Training_001/BraTS20_Training_001_seg.nii.gz"

mri_2.tumor_segmentation.set_compartment_masks()
"""

"""
run_wms = True
if run_wms:
    path = state_2.study_dir + of.helper.DER_DIR + state_2.subject + os.sep + state_2.dir + "wms" + os.sep
    working_folder = of.helper.general.mkdir_if_not_exist(path)
    structural_input_files = [mri_2.t1_dir]
    mri_2.set_wm_segmentation()
    mri_2.wm_segmentation.tumor_handling_approach = "bias_corrected"
    mri_2.wm_segmentation.set_input_wm_seg(structural_input_files)
    mri_2.wm_segmentation.run() 
