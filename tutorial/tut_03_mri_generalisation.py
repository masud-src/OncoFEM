"""

"""
import oncofem as of
import os

study = of.helper.structure.Study("tut_02")

subj_1 = study.create_subject("Subject_1")
state_1 = subj_1.create_state("init_state", datetime.date.today())
measure_1 = state_1.create_measure("data/Suditsch/T1", "t1")
measure_2 = state_1.create_measure("data/Suditsch/Flair", "flair")

mri = of.MRI(state_1)

mri.set_generalisation()
for measure in [measure_1, measure_2]:
    mri.generalisation.dcm2niigz(measure)
    mri.generalisation.bias_correction(measure)

print("begin coregister")
mri.generalisation.coregister_modality2atlas()   
print("begin skull strip")
mri.generalisation.skull_strip()

mri.t1_dir = measure_1.dir_act
mri.flair_dir = measure_2.dir_act

mri.generalisation.run_all()