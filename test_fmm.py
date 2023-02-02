# Imports
import os
from oncofem.helper.general import set_working_folder
from oncofem.struct.study import Study
from oncofem.struct.problem import Problem
from oncofem.mri.white_matter_segmentation import WhiteMatterSegmentation
from oncofem.helper import io
from oncofem.modelling.base_model.stochastic_model import Stochastic_Model

########################################################################################################################

#Define Study
study = Study("milestone")

# Defining of general Problem
x = Problem()

folder = "/media/marlon/data/MRI_data/UPENN-GBM/"
x.mri.t1_dir          = folder + "images_structural/UPENN-GBM-00002_11/UPENN-GBM-00002_11_T1.nii.gz"
x.mri.t1ce_dir        = folder + "images_structural/UPENN-GBM-00002_11/UPENN-GBM-00002_11_T1GD.nii.gz"
x.mri.t2_dir          = folder + "images_structural/UPENN-GBM-00002_11/UPENN-GBM-00002_11_T2.nii.gz"
x.mri.flair_dir       = folder + "images_structural/UPENN-GBM-00002_11/UPENN-GBM-00002_11_FLAIR.nii.gz"
x.mri.tumor_seg_dir   = folder + "automated_approx_segm.nii.gz"

st = Stochastic_Model()


# general info
st.get_init_distribution("/media/marlon/data/MRI_data/UPENN-GBM/images_segm/UPENN-GBM-00002_11_segm.nii.gz")
st.input_t1 = nib.load("/media/marlon/data/MRI_data/UPENN-GBM/images_segm/UPENN-GBM-00002_11_T1.nii.gz")
st.debug = False
st.id_edema = 2  # UPENN-GBM: 2
st.id_activ = 4  # UPENN-GBM: 4
st.id_necro = 1  # UPENN-GBM: 1
#st.id_unkno = 3  # BraTS2015: 3
T_end = 10
dt = 1

# growth characteristics
alpha_act = 0.5  # value from ratio
alpha_nec = 0.2  # value from ratio
alpha_ede = 2.0  # value from ratio

def linear_growth_activ():
    return alpha_act * st.vol_activ
st.growth_model_activ = linear_growth_activ
def l_g_n():
    return alpha_nec * (st.vol_activ / st.vol_necro)
st.growth_model_necro = l_g_n
def l_g_e():
    return alpha_ede * ((st.vol_activ+st.vol_necro) / st.vol_edema)
st.growth_model_edema = l_g_e

# set initial compartments
st.set_init_compartments()

# get skull mask
st.create_skull_border(st.input_t1)

# perform white matter segmentation
run_wms = False
if run_wms:
    working_folder = set_working_folder(working_folder + "wms" + os.sep)
    structural_input_files = [x.mri.t1_dir, x.mri.t2_dir]
    wms = WhiteMatterSegmentation(study)
    wms.set_input_wm_seg(structural_input_files, x.mri.tumor_seg_dir, work_dir=working_folder)
    wms.run_wm_seg(x) 

# perform dti segmentation


# perform dsc segmentation

# formulate preferred growth dir

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