"""
In this first tutorial, the general structure elements of OncoFEM are shown.

Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""
import oncofem as of
import datetime
import os

#############################################################################
# Creating a study
study = of.Study("tut_01")

#############################################################################
# Creating a subject and adding to study
subj_1 = study.create_subject("Subject_1")
# Alternatively this can be done separated
subj_2 = of.Subject("Subject_2")
subj_2.study_dir = study.dir
study.subjects.append(subj_2)

#############################################################################
# Creating states of a subject and adding this to subject and study
state_1 = subj_1.create_state("init_state", datetime.date.today())
# Alternatively this can be done separated
state_2 = of.State("evaluation_state", datetime.date.today())
state_2.subject = subj_1
state_2.study_dir = study.dir

#############################################################################
# Creating a measure of a subject at a certain state within a study
measure_1 = state_1.create_measure("tutorial/data/tut_01/BraTS20_Training_001_t1.nii", "t1")

measure_2 = of.Measure("tutorial/data/tut_01/BraTS20_Training_001_t1ce.nii", "t1ce")
state_1.measures.append(measure_2)

measure_3 = of.Measure("tutorial/data/tut_01/BraTS20_Training_001_t2.nii", "t2")
measure_4 = of.Measure("tutorial/data/tut_01/BraTS20_Training_001_flair.nii", "flair")
measure_5 = of.Measure("tutorial/data/tut_01/BraTS20_Training_001_seg.nii", "flair")
state_1.measures.append(measure_3)
state_1.measures.append(measure_4)
state_1.measures.append(measure_5)

#############################################################################
# Check if full modality
print(state_1.isFullModality())

#############################################################################
# Check for paths
print(state_1.t1_dir)
print(state_1.t1ce_dir)

#############################################################################
# Add paths manually
state_1.t1ce_dir = measure_2.dir_src
state_1.t2_dir = measure_3.dir_src
state_1.flair_dir = measure_4.dir_src
state_1.tumor_seg = measure_5.dir_src

mri = of.MRI(state_1)

mri.load_measures()

problem = of.Problem(mri=mri)
problem = of.Problem()

#############################################################################
# Generate geometry
geometry = of.Geometry()
title = "2D_QuarterCircle"
der_file = study.der_dir + title
der_path = der_file + os.sep
geometry.mesh, geometry.facet_function = of.geom.create_2D_QuarterCircle(10, 1.0, 1.0, 5, der_file, der_path)
geometry.dim = 2






