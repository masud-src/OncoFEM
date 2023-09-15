"""
paper stochastic model

# Definition of testcase
#
# Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
# 
# In this example the subject 2 of UPENN-GBM data set is modelled.
# First step in investigation is to define the study, the subject and the 
# state of measurement. This will create the output folders. Next step is the
# performed white matter segmentation. Herein, the csf and white and grey 
# matter is filtered. Than neccessary parameters are set and the model is
# initialised. Finally, the simulation is done.
#
# --------------------------------------------------------------------------#
"""

# Imports
import datetime
import os
import oncofem as of

##############################################################################
#Definition of input mri scans
study = of.Study("stochastic_model")
subj = study.create_subject("UPENN-GBM-00002")
state = subj.create_state("state_1", datetime.date.today())

folder = "/media/marlon/data/MRI_data/UPENN-GBM/images_structural/UPENN-GBM-00002_11/"
#folder = "/media/marlon/data/MRI_data/UPENN-GBM/images_segm/"
state.create_measure(folder + "UPENN-GBM-00002_11_T1.nii.gz", "t1")
state.create_measure(folder + "UPENN-GBM-00002_11_T1GD.nii.gz", "t1ce")
state.create_measure(folder + "UPENN-GBM-00002_11_T2.nii.gz", "t2")
state.create_measure(folder + "UPENN-GBM-00002_11_FLAIR.nii.gz", "flair")
folder = "/media/marlon/data/MRI_data/UPENN-GBM/images_segm/"
state.create_measure(folder + "UPENN-GBM-00002_11_segm.nii.gz", "seg")
##############################################################################

##############################################################################
# Processing of white matter
mr_unit = of.MRI(state)
mr_unit.load_measures()
mr_unit.wm_segmentation.set_input_wm_seg([mr_unit.t1_dir], mr_unit.tumor_seg, work_dir=study.der_dir+"wm_seg"+os.sep, modality="t1")
#mr_unit.wm_segmentation.run_all()
##############################################################################
mr_unit.wm_segmentation.tumor_dirs = ['/media/marlon/data/studies/stochastic_model/der/wm_seg/wms_Tumor_pve_0.nii.gz', 
                                      '/media/marlon/data/studies/stochastic_model/der/wm_seg/wms_Tumor_pve_1.nii.gz', 
                                      '/media/marlon/data/studies/stochastic_model/der/wm_seg/wms_Tumor_pve_2.nii.gz']
mr_unit.wm_segmentation.brain_dirs = ['/media/marlon/data/studies/stochastic_model/der/wm_seg/wms_Brain_pve_0.nii.gz', 
                                      '/media/marlon/data/studies/stochastic_model/der/wm_seg/wms_Brain_pve_1.nii.gz', 
                                      '/media/marlon/data/studies/stochastic_model/der/wm_seg/wms_Brain_pve_2.nii.gz']

##############################################################################
# Defining of general Problem
x = of.Problem(mr_unit)
x.param.gen.sol_dir = study.sol_dir + "stochastical_model" + os.sep
x.param.gen.debug = False
x.param.id_edema = 2  # UPENN-GBM: 2
x.param.id_activ = 4  # UPENN-GBM: 4
x.param.id_necro = 1  # UPENN-GBM: 1
#x.param.id_unkno = 3  # BraTS2015: 3
x.param.time.T_end = 50.0*24*3600.
x.param.time.dt = 24.*3600.  # in seconds
x.param.time.output_intervall = 24*3600.

##############################################################################
# initialize Stochastic Model
st = of.Stochastic_Model()

##############################################################################
# growth characteristics
alpha_act = 4e-7  # value from ratio
alpha_nec = 0.1e-7  # value from ratio
alpha_ede = 7.5e-5#1.0e4  # value from ratio
def linear_growth_activ():
    return alpha_act * st.vol_activ * st.dt
def l_g_n():
    return alpha_nec * (st.vol_activ / st.vol_necro) * st.dt
def l_g_e():
    return alpha_ede * ((st.vol_activ+st.vol_necro) / st.vol_edema) * st.dt

x.bmm.growth_model_activ = linear_growth_activ
x.bmm.growth_model_necro = l_g_n
x.bmm.growth_model_edema = l_g_e
##############################################################################
# Distribution characteristics
st.prop_growth_directions.fac_csf_b = 0.32
st.prop_growth_directions.fac_csf_t = 0.32
st.prop_growth_directions.fac_wm_b = 1.00
st.prop_growth_directions.fac_wm_t = 1.00
st.prop_growth_directions.fac_gm_b = 0.12
st.prop_growth_directions.fac_gm_t = 0.12

# Simulation
st.set_param(x)
st.initialize_model()
st.run_simulation()