"""
Handling of medical images

Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""

from oncofem.struc.state import State
from oncofem.mri.generalisation import Generalisation
from oncofem.mri.tumor_segmentation.tumor_segmentation import TumorSegmentation
from oncofem.mri.white_matter_segmentation import WhiteMatterSegmentation

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
        self.tumor_segmentation = TumorSegmentation
        self.wm_segmentation = None
        self.generalisation = Generalisation(self)   
        self.tumor_segmentation = TumorSegmentation(self)
        self.wm_segmentation = WhiteMatterSegmentation(self.state)

    def load_measures(self):
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