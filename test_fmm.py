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
class Stochastic_Model:

    def __init__(self):
        self.input_segm = None
        self.input_t1 = None
        self.input_t1Gd = None
        self.input_t2 = None
        self.input_flair = None
        self.input_dti = None
        self.input_dsc = None
        
        self.header = None
        self.affine = None
        self.sx = None
        self.sy = None
        self.sz = None
        self.volume = None
        
        self.id_edema = None
        self.id_activ = None
        self.id_necro = None
        self.id_inact = None
        
        self.bound_edema = None
        self.bound_activ = None
        self.bound_necro = None
        self.bound_inact = None
        
        self.distr_edema = None
        self.distr_activ = None
        self.distr_necro = None
        self.distr_inact = None
        
        self.vol_edema = None        
        self.vol_activ = None
        self.vol_necro = None
        self.vol_inact = None

        self.growth_vol_edema = None
        self.growth_vol_activ = None
        self.growth_vol_necro = None
        
        self.actual_step = None
        self.brain_border = None
        self.cwg_matter = None
        
        self.growth_model_edema = None
        self.growth_model_activ = None
        self.growth_model_necro = None

        self.T_end = None
        self.dt = None

        self.debug = None

    def get_init_distribution(self, direction):
        """
        Loads original nifti image from 'direction', generates a numpy array and safes header and affine for using the same space again, also get2know
        measures sx, sy, and sz and volume
        """
        self.input_segm = nib.load(direction)
        self.header = self.input_segm.header
        self.affine = self.input_segm.affine
        self.sx, self.sy, self.sz = self.header.get_zooms()
        self.volume = self.sx * self.sy * self.sz

    def create_mask(self, img_data, thres, out_val=1):
        """
        Creates mask, by setting every value higher than threshold to out_val
        """
        img_data[img_data > thres] = out_val
        return img_data.astype(int)

    def create_boundary_mask(self, img_data, thres: float, out_val: int, mode="outer"):
        """
        creates mask of boundary for object in given image
        """
        img_data[img_data > thres] = out_val  # set every value greater than 0 to 1
        return skimage.segmentation.find_boundaries(img_data, mode=mode)  # select all outer pixels to growth area

    def create_skull_border(self, orig_img):
        """
        Creates solid skull border around chosen image
        """
        image_data = st.get_fdata(orig_img)
        if st.debug:
            st.write_output_field(image_data, 0.0, "t1", "t1_raw")
        brain_mask = st.create_mask(image_data, 0, 1)
        if st.debug:
            st.write_output_field(brain_mask, 0.0, "t1", "brain_mask")
        st.brain_border = copy.deepcopy(st.create_boundary_mask(brain_mask, 0, 1))
        if st.debug:
           st.write_output_field(st.brain_border, 0.0, "t1", "brain_border")

    def create_structural_segmentation(self, images: list):
        """
        uses fsl fast algorithm
        """

    def calc_actual_volume(self, img_data):
        """
        calculates volume of given img_data and values between 0.0 and 1.0. Multiplies this value with pixel volume
        """
        coords = np.where((0.0 < img_data) & (img_data <= 1.0)) 
        return sum([self.volume * img_data[coords[0][x], coords[1][x], coords[2][x]] for x in range(len(coords[0]))])

    def get_fdata(self, orig_image, compartment=None, inner_compartments=None):
        """
        Gives deep copy of original image with selected compartments
        """
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

    def set_init_compartments(self):
        if self.id_edema is not None:
            self.distr_edema = self.get_fdata(self.input_segm, compartment=self.id_edema)
            if st.debug: st.write_output_field(st.distr_edema, 0.0, "edema", "init_edema")
        if self.id_activ is not None:
            self.distr_activ = self.get_fdata(self.input_segm, compartment=self.id_activ)
            if st.debug: st.write_output_field(st.distr_activ, 0.0, "activ", "init_active")
        if self.id_necro is not None:
            self.distr_necro = self.get_fdata(self.input_segm, compartment=self.id_necro)
            if st.debug: st.write_output_field(st.distr_necro, 0.0, "necro", "init_necro")
        if self.id_inact is not None:
            self.distr_inact = self.get_fdata(self.input_segm, compartment=self.id_inact)
            if st.debug: st.write_output_field(st.distr_inact, 0.0, "necro", "init_inact")
     
    def write_output_field(self, field, t, field_name: str, file_name: str, type="nii"):
        if type == "nii":
            img = nib.Nifti1Image(field, self.affine, self.header)
            nib.save(img, file_name + "_" + str(t) + ".nii")          
        elif type == "xdmf":
            # check with fieldmapgenerator
            # map_field(self, field_file, outfile, mesh_file=None)
            pass

    def get_growth_directions(self, coords, ):
        pass
    
    def growth_timestep(self, act_data, rest_vol, closed_vol):
        iter = 0
        while rest_vol > 0.0:
            iter = iter + 1
            # get2know boundary
            bound = self.find_boundary_cells(closed_vol)
            coord_bound = np.where(bound == 1)

            # get2know growth directions        
            # TODO: growth directions
            pref_dir = self.get_growth_directions(coord_bound)

            n_cells = np.shape(coord_bound)[1]
            full_vol = sum([act_data[coord_bound[0][x], coord_bound[1][x], coord_bound[2][x]] for x in range(len(coord_bound[0]))])
            free_vol = n_cells * 1.0 - full_vol  #volume
            if self.debug:
                nib.save(nib.Nifti1Image(bound, affine, header), "bound_" + str(iter) + ".nii")

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

    def run_simulation(self):
        # get the start time
        st = time.time()
        print("Begin simulation at ", st)
        
        t = 0.0
        while t < self.T_end:
            t += self.dt
            print("Timestep: ", t)
            # get2know growth kinematic

            # calculate volumes
            self.vol_activ = self.calc_actual_volume(self.activ_data)
            self.vol_necro = self.calc_actual_volume(self.necro_data)
            self.vol_edema = self.calc_actual_volume(self.edema_data)

            # calculate volumetric growth
            growth_activ = self.growth_model_activ()
            growth_necro = self.growth_model_necro()
            growth_edema = self.growth_model_edema()

            # set rest for while loop
            rest_edema = growth_edema + growth_activ + growth_necro
            rest_activ = growth_activ + growth_necro
            rest_necro = growth_necro

            edema = growth_timestep(edema_data, rest_edema, edema_data + activ_data + necro_data)
            ede_img = nib.Nifti1Image(edema, affine, header)
            nib.save(ede_img, "growth_ede_" + str(t) + ".nii")
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
            seg_img = nib.Nifti1Image(edema_mask + activ_mask + necro_mask, affine, header)
            nib.save(seg_img, "growth_seg_" + str(t) + ".nii")

        # get the end time
        et = time.time()
        # get the execution time
        elapsed_time = et - st
        print('Process time:', elapsed_time, 'seconds')
        
        pass
########################################################################################################################

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
images = [st.input_t1, st.input_t1Gd, st.input_t2, st.input_flair]
st.create_structural_segmentation(images)


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