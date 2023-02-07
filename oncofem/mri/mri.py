"""
# **************************************************************************#
#                                                                           #
# === MRI ==================================================================#
#                                                                           #
# **************************************************************************#
# Handling of medical images
#
# Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
#
# --------------------------------------------------------------------------#
"""

from oncofem.struc.state import State
from oncofem.mri.generalisation import Generalisation
from oncofem.mri.tumor_segmentation import TumorSegmentation
from oncofem.mri.white_matter_segmentation import WhiteMatterSegmentation
from oncofem.helper.io import write_output_field
from oncofem.helper.constant import DEBUG
import nibabel as nib
import skimage.segmentation
import copy
import numpy as np


# **************************************************************************#
#      Classes                                                              #
# **************************************************************************#
# Definition of Classes

class MRI:
    """
    t.b.d.
    """
    def __init__(self, state: State):
        self.study_dir = state.study_dir
        self.state = state
        self.t1_dir = None
        self.t1ce_dir = None
        self.t2_dir = None
        self.flair_dir = None
        self.tumor_seg_dir = None
        self.wm_seg_dir = None
        self.generalisation = Generalisation(self.state)
        self.tumor_segmentation = TumorSegmentation(self.state)
        self.wm_segmentation = WhiteMatterSegmentation(self.state)
        
    def get_nii_file(self, directory):
        return nib.load(directory)
    
    def load_measures(self):
        self.state.isFullModality()
        for measure in self.state.measures:
            if measure.modality == "t1":
                self.t1_dir = measure.dir_src
            if measure.modality == "t1ce":
                self.t1ce_dir = measure.dir_src
            if measure.modality == "t2":
                self.t2_dir = measure.dir_src
            if measure.modality == "flair":
                self.flair_dir = measure.dir_src
            if measure.modality == "seg":
                self.tumor_seg = measure.dir_src        

    def create_mask(self, img_data, thres: float, out_val=1, boundary=False, mode="outer"):
        """
        Creates mask, by setting every value higher than threshold to out_val. Optional only the border can be chosen.
        """
        img_data[img_data > thres] = out_val
        if boundary:
            return skimage.segmentation.find_boundaries(img_data, mode=mode)  # select all outer pixels to growth area
        else:
            return img_data.astype(int)

    def create_skull_border(self, orig_img):
        """
        Creates solid skull border around chosen image
        """
        image_data = self.get_fdata(orig_img)
        if DEBUG:
            write_output_field(image_data, 0.0, "t1", "debug_t1_raw")
        brain_mask = self.create_mask(image_data, 0, 1)
        if DEBUG:
            write_output_field(brain_mask, 0.0, "t1", "debug_brain_mask")
        self.brain_border = copy.deepcopy(self.create_mask(brain_mask, 0, 1, True))
        if DEBUG:
            write_output_field(self.brain_border, 0.0, "t1", "debug_brain_border")
            
    def get_fdata(self, orig_image, compartment=None, inner_compartments=None):
        """
        Gives deep copy of original image with selected compartments
        """
        mask = copy.deepcopy(orig_image)
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