"""
In this second tutorial, the segmentation of mri images is shown.

Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""
import oncofem as of
import datetime
import os

#############################################################################
# Creating a study
study = of.Study("tut_01")
subj_1 = study.create_subject("Subject_1")
state_1 = subj_1.create_state("init_state", datetime.date.today())
measure_1 = state_1.create_measure("tutorial/data/tut_01/BraTS20_Training_001_t1.nii", "t1")
measure_2 = state_1.create_measure("tutorial/data/tut_01/BraTS20_Training_001_t1ce.nii", "t1ce")
measure_3 = state_1.create_measure("tutorial/data/tut_01/BraTS20_Training_001_t2.nii", "t2")
measure_4 = state_1.create_measure("tutorial/data/tut_01/BraTS20_Training_001_flair.nii", "flair")