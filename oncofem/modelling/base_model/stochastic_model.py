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

from oncofem.modelling.base_model.base_model import BaseModel
from oncofem.helper.general import mkdir_if_not_exist
from oncofem.helper.io import write_field2nii
from oncofem.helper.constant import DEBUG
import skimage.segmentation
import copy
import numpy as np
import nibabel as nib
import time

########################################################################################################################

class Props_Pref_Dir:
    def __init__(self):
        self.brain_border = True
        self.wm_seg = True
        self.fac_csf_b = None
        self.fac_csf_t = None
        self.fac_wm_b = None
        self.fac_wm_t = None
        self.fac_gm_b = None
        self.fac_gm_t = None


class Stochastic_Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.input_segm = None
        self.input_t1 = None
        self.input_t1Gd = None
        self.input_t2 = None
        self.input_flair = None
        self.output_path = None

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

        self.bound_edema = None
        self.bound_activ = None
        self.bound_necro = None

        self.distr_edema = None
        self.distr_activ = None
        self.distr_necro = None

        self.vol_edema = None        
        self.vol_activ = None
        self.vol_necro = None

        self.growth_vol_edema = None
        self.growth_vol_activ = None
        self.growth_vol_necro = None

        self.actual_step = None
        self.prop_growth_directions = Props_Pref_Dir()
        self.growth_directions = None

        self.brain_mask = None
        self.csf_b_distr = None
        self.csf_t_distr = None
        self.wm_b_distr = None
        self.wm_t_distr = None
        self.gm_b_distr = None
        self.gm_t_distr = None

        self.growth_model_edema = None
        self.growth_model_activ = None
        self.growth_model_necro = None

        self.T_end = None
        self.dt = None
        self.output_intervall = None

        self.debug = None

    def set_param(self, input):
        self.output_path = input.param.gen.sol_dir
        self.id_edema = input.param.id_edema
        self.id_activ = input.param.id_activ
        self.id_necro = input.param.id_necro
        self.T_end = input.param.time.T_end
        self.dt = input.param.time.dt
        self.output_intervall = input.param.time.output_intervall
        self.debug = input.param.gen
        self.input_segm = input.mri.tumor_seg
        self.input_t1 = input.mri.t1_dir
        self.input_t1Gd = input.mri.t1ce_dir
        self.input_t2 = input.mri.t2_dir
        self.input_flair = input.mri.flair_dir
        self.wm_brain_dirs = input.mri.wm_segmentation.brain_dirs
        self.wm_tumor_dirs = input.mri.wm_segmentation.tumor_dirs
        self.growth_model_activ = input.bmm.growth_model_activ
        self.growth_model_edema = input.bmm.growth_model_edema
        self.growth_model_necro = input.bmm.growth_model_necro

    def initialize_model(self):
        """
        Loads original nifti image from 'direction', generates a numpy array and safes header and affine for using the same space again, also get2know
        measures sx, sy, and sz and volume
        """
        # make output path
        mkdir_if_not_exist(self.output_path)

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
            write_field2nii(self.distr_edema, 0.0, "edema", self.output_path + "init_edema", self.affine, self.header)
        if self.id_activ is not None:
            self.distr_activ = self.get_fdata(self.init_segm, compartment=self.id_activ)
            write_field2nii(self.distr_activ, 0.0, "activ", self.output_path + "init_active", self.affine, self.header)
        if self.id_necro is not None:
            self.distr_necro = self.get_fdata(self.init_segm, compartment=self.id_necro)
            write_field2nii(self.distr_necro, 0.0, "necro", self.output_path + "init_necro", self.affine, self.header)

        # Set up skull and stuff
        self.create_skull_border()

        # Set up growth directions
        self.set_growth_directions()

    def calc_actual_volume(self, img_data):
        """
        calculates volume of given img_data and values between 0.0 and 1.0. Multiplies this value with pixel volume
        """
        coords = np.where((0.0 < img_data) & (img_data <= 1.0)) 
        return sum([self.volume * img_data[coords[0][x], coords[1][x], coords[2][x]] for x in range(len(coords[0]))])

    def set_growth_directions(self):
        """
        Sets actual growth directions
        """
        self.growth_directions = copy.deepcopy(self.get_fdata(self.init_segm))
        self.growth_directions[:] = 1.0
        if self.prop_growth_directions.brain_border:
            self.growth_directions[:] = 0.0
            coords = np.where(self.brain_mask == 1)
            n_cells = len(coords[0])
            for i in range(n_cells):
                self.growth_directions[coords[0][i], coords[1][i], coords[2][i]] = 1.0
        if self.prop_growth_directions.wm_seg:
            self.gm_b_distr = copy.deepcopy(self.prop_growth_directions.fac_gm_b * self.get_fdata(nib.load(self.wm_brain_dirs[1])))
            self.gm_t_distr = copy.deepcopy(self.prop_growth_directions.fac_gm_t * self.get_fdata(nib.load(self.wm_tumor_dirs[1])))
            self.wm_b_distr = copy.deepcopy(self.prop_growth_directions.fac_wm_b * self.get_fdata(nib.load(self.wm_brain_dirs[2])))
            self.wm_t_distr = copy.deepcopy(self.prop_growth_directions.fac_wm_t * self.get_fdata(nib.load(self.wm_tumor_dirs[2])))
            self.csf_b_distr = copy.deepcopy(self.prop_growth_directions.fac_csf_b * self.get_fdata(nib.load(self.wm_brain_dirs[0])))
            self.csf_t_distr = copy.deepcopy(self.prop_growth_directions.fac_csf_t * self.get_fdata(nib.load(self.wm_tumor_dirs[0])))
            coords = np.where(self.growth_directions == 1)
            n_cells = len(coords[0])
            for i in range(n_cells):
                self.growth_directions[coords[0][i], coords[1][i], coords[2][i]] = self.gm_b_distr[coords[0][i], coords[1][i], coords[2][i]]
                self.growth_directions[coords[0][i], coords[1][i], coords[2][i]] += self.gm_t_distr[coords[0][i], coords[1][i], coords[2][i]]
                self.growth_directions[coords[0][i], coords[1][i], coords[2][i]] += self.wm_b_distr[coords[0][i], coords[1][i], coords[2][i]]
                self.growth_directions[coords[0][i], coords[1][i], coords[2][i]] += self.wm_t_distr[coords[0][i], coords[1][i], coords[2][i]]
                self.growth_directions[coords[0][i], coords[1][i], coords[2][i]] += self.csf_b_distr[coords[0][i], coords[1][i], coords[2][i]]
                self.growth_directions[coords[0][i], coords[1][i], coords[2][i]] += self.csf_t_distr[coords[0][i], coords[1][i], coords[2][i]]

        if DEBUG:
            write_field2nii(self.growth_directions, 0.0, "necro", self.output_path + "debug_growth_directions", self.affine, self.header)

        #elif self.type_growth_directions == "test_2area":
        #    t1 = self.get_fdata(self.input_t1)
        #    self.growth_directions = copy.deepcopy(t1)
        #    self.growth_directions[:] = 0.0
        #    coords = np.where(self.brain_mask == 1)
        #    n_cells = len(coords[0])
        #    for i in range(n_cells):
        #        if coords[2][i] > 80:
        #            self.growth_directions[coords[0][i], coords[1][i], coords[2][i]] = 1.0
        #        else:
        #            self.growth_directions[coords[0][i], coords[1][i], coords[2][i]] = 0.3

    def get_growth_directions(self, coords):        
        return self.growth_directions

    def growth_timestep(self, act_data, rest_vol, closed_vol):
        iter = 0
        while rest_vol > 0.0 and iter <= 30:
            iter = iter + 1
            # get2know boundary
            bound = self.create_mask((closed_vol).astype(int), 0.0, boundary=True)
            coords = np.where(bound == 1)
            if DEBUG:
                nib.save(nib.Nifti1Image(bound, self.affine, self.header), self.output_path + "bound_" + str(iter) + ".nii.gz")

            # get2know growth directions
            max_load = self.get_growth_directions(coords)
            if DEBUG and iter == 1:
                nib.save(nib.Nifti1Image(max_load, self.affine, self.header), self.output_path + "max_load_" + str(iter) + ".nii.gz")

            n_cells = len(coords[0])
            potential_vol = sum([max_load[coords[0][x], coords[1][x], coords[2][x]] for x in range(n_cells)])          
            full_vol = sum([act_data[coords[0][x], coords[1][x], coords[2][x]] for x in range(n_cells)])
            free_vol = potential_vol - full_vol  #volume
            full_border_fill = rest_vol / free_vol

            if full_border_fill < 1.0 and full_border_fill >= 0.0: # final iteration
                dump = 0.0
                for i in range(n_cells): # loop over every single border cell
                    if max_load[coords[0][i], coords[1][i], coords[2][i]] > full_border_fill:
                        if act_data[coords[0][i], coords[1][i], coords[2][i]] + full_border_fill >= 1.0:
                            act_data[coords[0][i], coords[1][i], coords[2][i]] = 1.0
                            dump -= 1.0 - act_data[coords[0][i], coords[1][i], coords[2][i]]
                            #closed_vol[coords[0][i], coords[1][i], coords[2][i]] = 1.0
                        else:
                            act_data[coords[0][i], coords[1][i], coords[2][i]] = act_data[coords[0][i], coords[1][i], coords[2][i]] + full_border_fill
                            dump -= full_border_fill
                    else:
                        if act_data[coords[0][i], coords[1][i], coords[2][i]] + max_load[coords[0][i], coords[1][i], coords[2][i]] >= 1.0:
                            act_data[coords[0][i], coords[1][i], coords[2][i]] = 1.0
                            dump -= 1.0 - act_data[coords[0][i], coords[1][i], coords[2][i]]
                            #closed_vol[coords[0][i], coords[1][i], coords[2][i]] = 1.0
                        else:
                            act_data[coords[0][i], coords[1][i], coords[2][i]] = act_data[coords[0][i], coords[1][i], coords[2][i]] + max_load[coords[0][i], coords[1][i], coords[2][i]] 
                            dump -= max_load[coords[0][i], coords[1][i], coords[2][i]]
                rest_vol = 0.0
            else:  # more iterations needed
                dump = 0.0
                for i in range(n_cells):
                    if act_data[coords[0][i], coords[1][i], coords[2][i]] + max_load[coords[0][i], coords[1][i], coords[2][i]] >= 1.0:
                        act_data[coords[0][i], coords[1][i], coords[2][i]] = 1.0
                        dump -= 1.0 - act_data[coords[0][i], coords[1][i], coords[2][i]]
                        closed_vol[coords[0][i], coords[1][i], coords[2][i]] = 1.0
                    else:
                        act_data[coords[0][i], coords[1][i], coords[2][i]] = act_data[coords[0][i], coords[1][i], coords[2][i]] + max_load[coords[0][i], coords[1][i], coords[2][i]]
                        dump -= max_load[coords[0][i], coords[1][i], coords[2][i]]
                rest_vol -= abs(dump)
        return act_data

    def run_simulation(self):
        # get the start time
        st = time.time()
        print("Begin simulation at ", st)

        out_count = 0.0
        t = 0.0
        while t < self.T_end:
            t += self.dt
            out_count += self.dt
            print("Timestep: ", t, " in days: ", t / (24. * 3600.))
            # get2know growth kinematic

            # calculate volumes
            self.vol_activ = self.calc_actual_volume(self.distr_activ)
            self.vol_necro = self.calc_actual_volume(self.distr_necro)
            self.vol_edema = self.calc_actual_volume(self.distr_edema)

            # calculate volumetric growth
            self.growth_vol_activ = self.growth_model_activ()
            self.growth_vol_necro = self.growth_model_necro()
            self.growth_vol_edema = self.growth_model_edema()

            # set rest for while loop
            rest_edema = self.growth_vol_edema + self.growth_vol_activ + self.growth_vol_necro
            rest_activ = self.growth_vol_activ + self.growth_vol_necro
            rest_necro = self.growth_vol_necro

            edema = self.growth_timestep(self.distr_edema, rest_edema, copy.deepcopy(self.distr_edema + self.distr_activ + self.distr_necro))
            activ = self.growth_timestep(self.distr_activ, rest_activ, copy.deepcopy(self.distr_activ + self.distr_necro))
            necro = self.growth_timestep(self.distr_necro, rest_necro, copy.deepcopy(self.distr_necro))

            print("vol_edema: ", self.vol_edema, " rest_edema: ", rest_edema, " sum: ", self.vol_edema+rest_edema, 
                  " new vol: ", self.calc_actual_volume(edema), " delta: ", self.vol_edema+rest_edema - self.calc_actual_volume(edema), 
                  " relative: ", (self.vol_edema+rest_edema - self.calc_actual_volume(edema)) / max(self.vol_edema+rest_edema, self.calc_actual_volume(edema)) * 100.0)
            #print("vol_edema: ", self.vol_edema, " growth_vol: ", self.growth_vol_edema, " sum: ", self.vol_edema + self.growth_vol_edema,
            #      " new vol: ", self.calc_actual_volume(edema), " delta: ", self.vol_edema + self.growth_vol_edema - self.calc_actual_volume(edema),
            #      " relative: ", (
            #                  self.vol_edema + self.growth_vol_edema - self.calc_actual_volume(edema)) / max(self.vol_edema + self.growth_vol_edema, self.calc_actual_volume(edema)) * 100.0)

            if out_count >= self.output_intervall:  # Abfrage ob output
                out_count = 0.0
                ede_img = nib.Nifti1Image(edema, self.affine, self.header)
                nib.save(ede_img, self.output_path + "growth_ede_" + str(t) + ".nii.gz")
                act_img = nib.Nifti1Image(activ, self.affine, self.header)
                nib.save(act_img, self.output_path + "growth_act_" + str(t) + ".nii.gz")
                nec_img = nib.Nifti1Image(necro, self.affine, self.header)
                nib.save(nec_img, self.output_path + "growth_nec_" + str(t) + ".nii.gz")
                edema_mask = copy.deepcopy(edema)
                activ_mask = copy.deepcopy(activ)
                necro_mask = copy.deepcopy(necro)
                edema_mask[edema_mask > 0.0] = 1
                activ_mask[activ_mask > 0.0] = 2
                necro_mask[necro_mask > 0.0] = 3
                seg_img = nib.Nifti1Image(edema_mask + activ_mask + necro_mask, self.affine, self.header)
                nib.save(seg_img, self.output_path + "growth_seg_" + str(t) + ".nii.gz")

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
            write_field2nii(image_data, 0.0, "t1", self.output_path + "debug_t1_raw", self.affine, self.header)
        self.brain_mask = self.create_mask(image_data, 0, 1)
        if DEBUG:
            write_field2nii(self.brain_mask, 0.0, "t1", self.output_path + "debug_brain_mask", self.affine, self.header)

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
