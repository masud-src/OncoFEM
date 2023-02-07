# Imports
import datetime
import os
import oncofem as of

##############################################################################
#Definition of input mri scans
study = of.Study("stochastic_model")
subj = study.create_subject("UPENN-GBM-00002")
state = subj.create_state("state_1", datetime.date.today())

#folder = "/media/marlon/data/MRI_data/UPENN-GBM/"
folder = "/media/marlon/data/MRI_data/UPENN-GBM/images_segm/"
state.create_measure(folder + "UPENN-GBM-00002_11_T1.nii.gz", "t1")
state.create_measure(folder + "UPENN-GBM-00002_11_T1GD.nii.gz", "t1ce")
state.create_measure(folder + "UPENN-GBM-00002_11_T2.nii.gz", "t2")
state.create_measure(folder + "UPENN-GBM-00002_11_FLAIR.nii.gz", "flair")
state.create_measure(folder + "UPENN-GBM-00002_11_segm.nii.gz", "seg")
##############################################################################

##############################################################################
# Processing of white matter
mr_unit = of.MRI(state)
mr_unit.load_measures()
mr_unit.wm_segmentation.set_input_wm_seg([state.t1_dir], state.tumor_seg, work_dir=study.der_dir+"wm_seg"+os.sep, modality="t1")
#mr_unit.wm_segmentation.run_all()
##############################################################################
mr_unit.wm_segmentation.tumor_dirs = ['/media/marlon/data/studies/stochastic_model/der/wm_seg/wms_Tumor_pve_0', '/media/marlon/data/studies/stochastic_model/der/wm_seg/wms_Tumor_pve_1', '/media/marlon/data/studies/stochastic_model/der/wm_seg/wms_Tumor_pve_2']
mr_unit.wm_segmentation.brain_dirs = ['/media/marlon/data/studies/stochastic_model/der/wm_seg/wms_Brain_pve_0', '/media/marlon/data/studies/stochastic_model/der/wm_seg/wms_Brain_pve_1', '/media/marlon/data/studies/stochastic_model/der/wm_seg/wms_Brain_pve_2']

##############################################################################
# Defining of general Problem
x = of.Problem(mr_unit)
x.param.gen.debug = False
x.param.id_edema = 2  # UPENN-GBM: 2
x.param.id_activ = 4  # UPENN-GBM: 4
x.param.id_necro = 1  # UPENN-GBM: 1
#x.param.id_unkno = 3  # BraTS2015: 3
x.param.T_end = 10.0
x.param.dt = 1.0

##############################################################################




# formulate preferred growth dir


st = of.Stochastic_Model()
# set initial compartments
st.set_init_compartments()
# get skull mask
st.create_skull_border(st.input_t1)

# general info
st.get_init_distribution(state.tumor_seg_dir)
st.input_t1 = x.mri.get_nii_file("/media/marlon/data/MRI_data/UPENN-GBM/images_segm/UPENN-GBM-00002_11_T1.nii.gz")
st.debug = False
st.id_edema = 2  # UPENN-GBM: 2

##############################################################################
# growth characteristics
alpha_act = 0.5  # value from ratio
alpha_nec = 0.2  # value from ratio
alpha_ede = 2.0  # value from ratio
def linear_growth_activ():
    return alpha_act * st.vol_activ
def l_g_n():
    return alpha_nec * (st.vol_activ / st.vol_necro)
def l_g_e():
    return alpha_ede * ((st.vol_activ+st.vol_necro) / st.vol_edema)

st.growth_model_activ = linear_growth_activ
st.growth_model_necro = l_g_n
st.growth_model_edema = l_g_e
##############################################################################




"""
# get the start time
st = time.time()

t = 0.0
while t < T_end:
    t += dt
    print("Timestep: ", t)
    ####################################################################
    # get2know growth kinematic

    # calculate volumes
    vol_activ = calc_actual_volume(activ_data)
    vol_necro = calc_actual_volume(necro_data)
    vol_edema = calc_actual_volume(edema_data)

    # calculate volumetric growth
    growth_activ = alpha_act * vol_activ + ground_growth
    growth_necro = alpha_nec * (vol_activ / vol_necro)
    growth_edema = alpha_ede * ((vol_activ+vol_necro) / vol_edema)

    # set rest for while loop
    rest_edema = growth_edema + growth_activ + growth_necro
    rest_activ = growth_activ + growth_necro
    rest_necro = growth_necro

    edema = growth_timestep(edema_data, rest_edema, edema_data+activ_data+necro_data)
    ede_img = nib.Nifti1Image(edema, affine, header)
    nib.save(ede_img, "growth_ede_"+str(t)+".nii")
    activ = growth_timestep(activ_data, rest_activ, activ_data + necro_data)
    act_img = nib.Nifti1Image(activ, affine, header)
    nib.save(act_img, "growth_act_" + str(t) + ".nii")
    necro = growth_timestep(necro_data, rest_necro, necro_data)
    nec_img = nib.Nifti1Image(necro, affine, header)
    nib.save(nec_img, "growth_nec_" + str(t) + ".nii")
    edema_mask = copy.deepcopy(edema)
    activ_mask = copy.deepcopy(activ)
    necro_mask = copy.deepcopy(necro)
    edema_mask[edema_mask > 0.0] = 1
    activ_mask[activ_mask > 0.0] = 2
    necro_mask[necro_mask > 0.0] = 3
    seg_img = nib.Nifti1Image(edema_mask+activ_mask+necro_mask, affine, header)
    nib.save(seg_img, "growth_seg_" + str(t) + ".nii")

# get the end time
et = time.time()
# get the execution time
elapsed_time = et - st
print('Process time:', elapsed_time, 'seconds')
"""