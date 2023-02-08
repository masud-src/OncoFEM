"""
# **************************************************************************#
#                                                                           #
# === Stochastic Model =====================================================#
#                                                                           #
# **************************************************************************#
# Definition of stochastical model
#
# Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
#
# --------------------------------------------------------------------------#
"""

from oncofem.helper.io import write_field2nii
from oncofem.helper.constant import DEBUG
import skimage.segmentation
import copy
import numpy as np
import nibabel as nib
import time

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

        self.init_segm = None
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
        self.cfs_distr = None
        self.wm_distr = None
        self.gm_distr = None

        self.growth_model_edema = None
        self.growth_model_activ = None
        self.growth_model_necro = None

        self.T_end = None
        self.dt = None

        self.debug = None

    def set_param(self, input):
        self.id_edema = input.param.id_edema
        self.id_activ = input.param.id_activ
        self.id_necro = input.param.id_necro
        self.T_end = input.param.time.T_end
        self.dt = input.param.time.dt
        self.debug = input.param.gen
        self.input_segm = input.mri.tumor_seg
        self.input_t1 = input.mri.t1_dir
        self.input_t1Gd = input.mri.t1ce_dir
        self.input_t2 = input.mri.t2_dir
        self.input_flair = input.mri.flair_dir
        self.growth_model_activ = input.bmm.growth_model_activ
        self.growth_model_edema = input.bmm.growth_model_edema
        self.growth_model_necro = input.bmm.growth_model_necro


    def initialize_model(self):
        """
        Loads original nifti image from 'direction', generates a numpy array and safes header and affine for using the same space again, also get2know
        measures sx, sy, and sz and volume
        """
        # Get2know principle informations
        self.init_segm = nib.load(self.input_segm)
        self.input_t1 = nib.load(self.input_t1)
        self.input_t1Gd = nib.load(self.input_t1Gd)
        self.input_t2 = nib.load(self.input_t2)
        self.input_flair = nib.load(self.input_flair)
        self.header = self.init_segm.header
        self.affine = self.init_segm.affine
        self.sx, self.sy, self.sz = self.header.get_zooms()
        self.volume = self.sx * self.sy * self.sz

        # Set up distribution of tumour constituents
        if self.id_edema is not None:
            self.distr_edema = self.get_fdata(self.init_segm, compartment=self.id_edema)
            if DEBUG:
                write_field2nii(self.distr_edema, 0.0, "edema", "debug_init_edema", self.affine, self.header)
        if self.id_activ is not None:
            self.distr_activ = self.get_fdata(self.init_segm, compartment=self.id_activ)
            if DEBUG:
                write_field2nii(self.distr_activ, 0.0, "activ", "debug_init_active", self.affine, self.header)
        if self.id_necro is not None:
            self.distr_necro = self.get_fdata(self.init_segm, compartment=self.id_necro)
            if DEBUG:
                write_field2nii(self.distr_necro, 0.0, "necro", "debug_init_necro", self.affine, self.header)
        if self.id_inact is not None:
            self.distr_inact = self.get_fdata(self.init_segm, compartment=self.id_inact)
            if DEBUG:
                write_field2nii(self.distr_inact, 0.0, "necro", "debug_init_inact", self.affine, self.header)

        # Set up skull and stuff
        self.create_skull_border()
        if DEBUG:
            write_field2nii(self.brain_border, 0.0, "necro", "debug_brain_border", self.affine, self.header)

    def calc_actual_volume(self, img_data):
        """
        calculates volume of given img_data and values between 0.0 and 1.0. Multiplies this value with pixel volume
        """
        coords = np.where((0.0 < img_data) & (img_data <= 1.0)) 
        return sum([self.volume * img_data[coords[0][x], coords[1][x], coords[2][x]] for x in range(len(coords[0]))])

    def get_growth_directions(self, coords, ):
        pass

    def growth_timestep(self, act_data, rest_vol, closed_vol):
        iter = 0
        while rest_vol > 0.0:
            iter = iter + 1
            # get2know boundary
            bound = self.create_mask(closed_vol.astype(int), 0.0, boundary=True)
            coords = np.where(bound == 1)
            coord_bound = []
            coord_bound = [coord_bound.append([coords[0][x], coords[1][x], coords[2][x]]) for x in range(len(coords[0]))]
            if DEBUG:
                nib.save(nib.Nifti1Image(bound, self.affine, self.header), "bound_" + str(iter) + ".nii")

            # get2know growth directions
            pref_dir = self.get_growth_directions(coord_bound)

            n_cells = np.shape(coord_bound)[1]
            potential_vol = sum([pref_dir[coord_bound[0][x], coord_bound[1][x], coord_bound[2][x]] for x in range(len(coord_bound[0]))])          
            full_vol = sum([act_data[coord_bound[0][x], coord_bound[1][x], coord_bound[2][x]] for x in range(len(coord_bound[0]))])
            free_vol = potential_vol - full_vol  #volume

            if rest_vol / free_vol < 1.0:
                for i in range(n_cells):
                    if rest_vol / free_vol > pref_dir[coord_bound[0][i], coord_bound[1][i], coord_bound[2][i]]:
                        act_data[coord_bound[0][i], coord_bound[1][i], coord_bound[2][i]] = act_data[coord_bound[0][i], coord_bound[1][i], coord_bound[2][i]] + pref_dir[coord_bound[0][i], coord_bound[1][i], coord_bound[2][i]]
                    act_data[coord_bound[0][i], coord_bound[1][i], coord_bound[2][i]] = rest_vol / free_vol
                rest_vol = 0.0
            elif rest_vol / free_vol >= 1.0:
                for i in range(n_cells):
                    rest_vol -= 1.0 - act_data[coord_bound[0][i], coord_bound[1][i], coord_bound[2][i]]
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
            self.vol_activ = self.calc_actual_volume(self.distr_activ)
            self.vol_necro = self.calc_actual_volume(self.distr_necro)
            self.vol_edema = self.calc_actual_volume(self.distr_edema)

            # calculate volumetric growth
            growth_activ = self.growth_model_activ()
            growth_necro = self.growth_model_necro()
            growth_edema = self.growth_model_edema()

            # set rest for while loop
            rest_edema = growth_edema + growth_activ + growth_necro
            rest_activ = growth_activ + growth_necro
            rest_necro = growth_necro

            edema = self.growth_timestep(self.distr_edema, rest_edema, self.distr_edema + self.distr_activ + self.distr_necro)
            activ = self.growth_timestep(self.distr_activ, rest_activ, self.distr_activ + self.distr_necro)
            necro = self.growth_timestep(self.distr_necro, rest_necro, copy.deepcopy(self.distr_necro))

            if True:  # Abfrage ob output
                ede_img = nib.Nifti1Image(edema, self.affine, self.header)
                nib.save(ede_img, "growth_ede_" + str(t) + ".nii")
                act_img = nib.Nifti1Image(activ, self.affine, self.header)
                nib.save(act_img, "growth_act_" + str(t) + ".nii")
                nec_img = nib.Nifti1Image(necro, self.affine, self.header)
                nib.save(nec_img, "growth_nec_" + str(t) + ".nii")
                edema_mask = copy.deepcopy(edema)
                activ_mask = copy.deepcopy(activ)
                necro_mask = copy.deepcopy(necro)
                edema_mask[edema_mask > 0.0] = 1
                activ_mask[activ_mask > 0.0] = 2
                necro_mask[necro_mask > 0.0] = 3
                seg_img = nib.Nifti1Image(edema_mask + activ_mask + necro_mask, self.affine, self.header)
                nib.save(seg_img, "growth_seg_" + str(t) + ".nii")

        # get the end time
        et = time.time()
        # get the execution time
        elapsed_time = et - st
        print('Process time:', elapsed_time, 'seconds')


    def create_mask(self, img_data, thres: float, out_val=1, boundary=False, mode="outer"):
        """
        Creates mask, by setting every value higher than threshold to out_val. Optional only the border can be chosen.
        """
        img_data[img_data > thres] = int(out_val)
        if boundary:
            return skimage.segmentation.find_boundaries(img_data, mode=mode)  # select all outer pixels to growth area
        else:
            return img_data.astype(int)

    def create_skull_border(self):
        """
        Creates solid skull border around chosen image
        """
        image_data = self.get_fdata(self.input_t1)
        if DEBUG:
            write_field2nii(image_data, 0.0, "t1", "debug_t1_raw", self.affine, self.header)
        brain_mask = self.create_mask(image_data, 0, 1)
        if DEBUG:
            write_field2nii(brain_mask, 0.0, "t1", "debug_brain_mask", self.affine, self.header)
        self.brain_border = copy.deepcopy(self.create_mask(brain_mask, 0, 1, True))
        if DEBUG:
            write_field2nii(self.brain_border, 0.0, "t1", "debug_brain_border", self.affine, self.header)

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