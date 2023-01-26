import nibabel as nib
import nibabel.nifti1
import numpy as np
import copy
import skimage.segmentation
import os
import SimpleITK as sitk
from skimage import data, util, measure
import pandas as pd
import time


########################################################################################################################
# create growth of boundary cells
def find_boundary_cells(img_data):
    edema_mask = copy.deepcopy(img_data)
    edema_mask[img_data >= 1.0] = 1  # set every value greater than 0 to 1
    boundary = edema_mask.astype(int)  # set to integer
    boundary[boundary > 1] = 1  # set every value to 1
    return skimage.segmentation.find_boundaries(boundary, mode="outer")  # select all outer pixels to growth area
########################################################################################################################
# Calculate actual volumes
def calc_actual_volume(img_data):
    coords = np.where((0.0 < img_data) & (img_data <= 1.0)) 
    return sum([img_data[coords[0][x], coords[1][x], coords[2][x]] for x in range(len(coords[0]))])

########################################################################################################################
def file_collector(path, ending=None):
    """
    Collects files in folders and subfolders with optional ending
    """
    for root, dirs, filenames in os.walk(path):
        for filename in filenames:
            if ending is None:
                yield os.path.join(root, filename)
            elif filename.endswith(ending):
                yield os.path.join(root, filename)

########################################################################################################################
def get_fdata(orig_image, compartment=None, inner_compartments=None):
    mask = copy.deepcopy(orig_image.get_fdata())
    unique = list(np.unique(mask))
    if compartment==None:
        return mask
    else:
        unique.remove(compartment)
        for outer in unique:
            mask[np.isclose(mask, outer)] = 0.0

        mask[np.isclose(mask, compartment)] = 1.0
        if inner_compartments is not None:
            for comp in inner_compartments:
                mask[np.isclose(mask, comp)] = 1.0
                unique.remove(comp)
    return mask
########################################################################################################################
def growth_timestep(act_data, rest_vol, closed_vol):
    iter = 0
    while rest_vol > 0.0:
        iter = iter + 1
        # get2know boundary
        bound = find_boundary_cells(closed_vol)
        coord_bound = np.where(bound == 1)
        n_cells = np.shape(coord_bound)[1]
        full_vol = sum([act_data[coord_bound[0][x], coord_bound[1][x], coord_bound[2][x]] for x in range(len(coord_bound[0]))])
        free_vol = n_cells * 1.0 - full_vol  #volume
        if debug:
            nib.save(nib.Nifti1Image(bound, affine, header), "bound_" + str(iter) + ".nii")

        # get2know growth directions        
        # TODO: growth directions

        if rest_vol / free_vol < 1.0:
            for i in range(n_cells):
                act_data[coord_bound[0][i], coord_bound[1][i], coord_bound[2][i]] = rest_vol / free_vol
            rest_vol = 0.0
        elif rest_vol / free_vol >= 1.0:
            rest_vol = rest_vol - free_vol
            for i in range(n_cells):
                act_data[coord_bound[0][i], coord_bound[1][i], coord_bound[2][i]] = 1.0
                closed_vol[coord_bound[0][i], coord_bound[1][i], coord_bound[2][i]] = 1.0
    return act_data

########################################################################################################################
debug=False
# get the start time
st = time.time()
########################################################################################################################
id_edema = 2  # UPENN-GBM: 2
id_activ = 4  # UPENN-GBM: 4
id_necro = 1  # UPENN-GBM: 1
id_unkno = 3  # BraTS2015: 3
orig_img = nib.load("/media/marlon/data/MRI_data/UPENN-GBM/images_segm/UPENN-GBM-00002_11_segm.nii.gz")
header = orig_img.header
affine = orig_img.affine
T_end = 10
dt = 1

ground_growth = 0.0
alpha_act = 0.5  # value from ratio
alpha_nec = 0.2  # value from ratio
alpha_ede = 2.0  # value from ratio

edema_data = get_fdata(orig_img, compartment=id_edema)
necro_data = get_fdata(orig_img, compartment=id_necro)
activ_data = get_fdata(orig_img, compartment=id_activ)
segme_data = get_fdata(orig_img)


if debug:
    ede_img = nib.Nifti1Image(edema_data, orig_img.affine, orig_img.header)
    nib.save(ede_img, "edema.nii")
    nec_img = nib.Nifti1Image(necro_data, orig_img.affine, orig_img.header)
    nib.save(nec_img, "necro.nii")
    act_img = nib.Nifti1Image(activ_data, orig_img.affine, orig_img.header)
    nib.save(act_img, "activ.nii")
    seg_img = nib.Nifti1Image(segme_data, orig_img.affine, orig_img.header)
    nib.save(seg_img, "segme.nii")

# get the preprocess time
et = time.time()
elapsed_time = et - st
print('Preprocess time:', elapsed_time, 'seconds')
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
    #ede_img = nib.Nifti1Image(edema, affine, header)
    #nib.save(ede_img, "growth_ede_"+str(t)+".nii")
    activ = growth_timestep(activ_data, rest_activ, activ_data + necro_data)
    #act_img = nib.Nifti1Image(activ, affine, header)
    #nib.save(act_img, "growth_act_" + str(t) + ".nii")
    necro = growth_timestep(necro_data, rest_necro, necro_data)
    #nec_img = nib.Nifti1Image(necro, affine, header)
    #nib.save(nec_img, "growth_nec_" + str(t) + ".nii")
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


#sx, sy, sz = header.get_zooms()
#volume = sx * sy * sz