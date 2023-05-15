"""
Handling of medical images

Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""

from oncofem.struc.state import State
from oncofem.mri.generalisation import Generalisation
from oncofem.mri.tumor_segmentation.tumor_segmentation import TumorSegmentation
from oncofem.mri.white_matter_segmentation import WhiteMatterSegmentation
import nibabel as nib
import fsl
import copy
import numpy as np

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
        self.full_ana_modality = None
        self.generalisation = None
        self.affine = None
        self.shape = None
        self.ede_mask = None
        self.act_mask = None
        self.nec_mask = None
        self.generalisation = Generalisation(self)   
        self.tumor_segmentation = TumorSegmentation(self)
        self.wm_segmentation = WhiteMatterSegmentation(self)

    def load_measures(self):
        image = nib.load(self.state.measures[0].dir_act)
        self.affine = image.affine
        self.shape = image.shape
        for measure in self.state.measures:
            if measure.modality == "t1":
                self.t1_dir = measure.dir_act
            if measure.modality == "t1ce":
                self.t1ce_dir = measure.dir_act
            if measure.modality == "t2":
                self.t2_dir = measure.dir_act
            if measure.modality == "flair":
                self.flair_dir = measure.dir_act
            if measure.modality == "seg":
                self.tumor_seg_dir = measure.dir_act        

    def isFullModality(self):
        list_available_modality = [measure.modality for measure in self.state.measures]
        list_full_modality = ["t1", "t1ce", "t2", "flair"]
        self.full_ana_modality = all(item in list_available_modality for item in list_full_modality)
        return self.full_ana_modality

def image2array(image_dir):
    """
    Takes a directory of an image and gives a numpy array
    """
    orig_image = nib.load(image_dir)
    return copy.deepcopy(orig_image.get_fdata()), orig_image.shape, orig_image.affine

def image2mask(image_dir, compartment=None, inner_compartments=None):
    """
    Gives deep copy of original image with selected compartments
    """
    mask, _, _ = image2array(image_dir)
    unique = list(np.unique(mask))
    unique.remove(compartment)
    for outer in unique:
        mask[np.isclose(mask, outer)] = 0.0
    mask[np.isclose(mask, compartment)] = 1.0
    if inner_compartments is not None:
        for comp in inner_compartments:
            mask[np.isclose(mask, comp)] = 1.0
            unique.remove(comp)
    return mask

def cut_area_from_image(input_image, area_mask, inverse=False):
    """
    tbd
    """
    if inverse:
        area_mask = fsl.wrappers.fslmaths(area_mask).mul(-1).add(1).run()

    return fsl.wrappers.fslmaths(input_image).mul(area_mask).run()
