"""
In this second tutorial, the segmentation of mri images is shown.

Hierbei zunächst an Suditsch brain gezeigt, wie Generalisierung funktioniert.
Tumor Segmentierung mit modal agnostic type.
White Matter segmentierung.

Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""
import oncofem as of
import datetime
import os
"""

"""
study = of.Study("tut_02")
subj_1 = study.create_subject("Subject_1")
state_1 = subj_1.create_state("init_state", datetime.date.today())
measure_1 = state_1.create_measure("tutorial/data/Suditsch/T1", "t1")
measure_2 = state_1.create_measure("tutorial/data/Suditsch/Flair", "flair")

mri = of.MRI(state_1)
mri.load_measures()
mri.generalisation.d2n.print_command = True

for measure in [measure_1, measure_2]:
    mri.generalisation.dcm2niigz(measure)
    mri.generalisation.bias_correction(measure)

print("begin coregister")
mri.generalisation.coregister_modality2atlas()   
print("begin skull strip")
mri.generalisation.skull_strip()

subj_2 = study.create_subject("Subject_2")
state_2 = subj_2.create_state("init_state")
measure_21 = state_2.create_measure("tutorial/data/BraTS20_Training_001/BraTS20_Training_001_t1.nii", "t1")
measure_22 = state_2.create_measure("tutorial/data/BraTS20_Training_001/BraTS20_Training_001_t1ce.nii", "t1ce")
measure_23 = state_2.create_measure("tutorial/data/BraTS20_Training_001/BraTS20_Training_001_t2.nii", "t2")
measure_24 = state_2.create_measure("tutorial/data/BraTS20_Training_001/BraTS20_Training_001_flair.nii", "flair")