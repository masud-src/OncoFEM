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
import nibabel as nib

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
