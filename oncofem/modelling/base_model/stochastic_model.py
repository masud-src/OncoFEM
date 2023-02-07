import nibabel as nib
import numpy as np
import copy
import time
from oncofem.mri.mri import MRI

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
        self.cfs_distr = None
        self.wm_distr = None
        self.gm_distr = None

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

    def calc_actual_volume(self, img_data):
        """
        calculates volume of given img_data and values between 0.0 and 1.0. Multiplies this value with pixel volume
        """
        coords = np.where((0.0 < img_data) & (img_data <= 1.0)) 
        return sum([self.volume * img_data[coords[0][x], coords[1][x], coords[2][x]] for x in range(len(coords[0]))])

    def set_init_compartments(self):
        if self.id_edema is not None:
            self.distr_edema = self.get_fdata(self.input_segm, compartment=self.id_edema)
            if self.debug: 
                self.write_output_field(self.distr_edema, 0.0, "edema", "debug_init_edema")
        if self.id_activ is not None:
            self.distr_activ = self.get_fdata(self.input_segm, compartment=self.id_activ)
            if self.debug: 
                self.write_output_field(self.distr_activ, 0.0, "activ", "debug_init_active")
        if self.id_necro is not None:
            self.distr_necro = self.get_fdata(self.input_segm, compartment=self.id_necro)
            if self.debug: 
                self.write_output_field(self.distr_necro, 0.0, "necro", "debug_init_necro")
        if self.id_inact is not None:
            self.distr_inact = self.get_fdata(self.input_segm, compartment=self.id_inact)
            if self.debug: 
                self.write_output_field(self.distr_inact, 0.0, "necro", "debug_init_inact")

    def get_growth_directions(self, coords, ):
        pass

    def growth_timestep(self, act_data, rest_vol, closed_vol):
        iter = 0
        while rest_vol > 0.0:
            iter = iter + 1
            # get2know boundary
            bound = self.create_mask(closed_vol, 0.0, boundary=True)
            coord_bound = np.where(bound == 1)

            # get2know growth directions        
            # TODO: growth directions
            pref_dir = self.get_growth_directions(coord_bound)

            n_cells = np.shape(coord_bound)[1]
            full_vol = sum([act_data[coord_bound[0][x], coord_bound[1][x], coord_bound[2][x]] for x in range(len(coord_bound[0]))])
            free_vol = n_cells * 1.0 - full_vol  #volume
            if self.debug:
                nib.save(nib.Nifti1Image(bound, self.affine, self.header), "bound_" + str(iter) + ".nii")

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

            edema = self.growth_timestep(self.distr_edema, rest_edema, copy.deepcopy(self.distr_edema + self.distr_activ + self.distr_necro))
            ede_img = nib.Nifti1Image(edema, self.affine, self.header)
            nib.save(ede_img, "growth_ede_" + str(t) + ".nii")
            activ = self.growth_timestep(self.distr_activ, rest_activ, copy.deepcopy(self.distr_activ + self.distr_necro))
            act_img = nib.Nifti1Image(activ, self.affine, self.header)
            nib.save(act_img, "growth_act_" + str(t) + ".nii")
            necro = self.growth_timestep(self.distr_necro, rest_necro, copy.deepcopy(self.distr_necro))
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
