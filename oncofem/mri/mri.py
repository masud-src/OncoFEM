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
from oncofem.interfaces.dcm2niix import Dcm2niix
import nibabel as nib


# **************************************************************************#
#      Classes                                                              #
# **************************************************************************#
# Definition of Classes

class MRI:
    """
    t.b.d.
    """

    def __init__(self):
        self.t1_dir = None
        self.t1ce_dir = None
        self.t2_dir = None
        self.flair_dir = None
        self.tumor_seg_dir = None
        self.wm_seg_dir = None
        self.dti_ad_dir = None
        self.dti_fa_dir = None
        self.dti_rd_dir = None
        self.dti_tr_dir = None
        self.dsc_psr_dir =None
        self.dsc_ph_dir = None
        self.dsc_ap_dir = None
        self.list_full_modality = ["T1", "T1CE", "T2", "FL"]
        self.inter_dcm2niix = Dcm2niix()
        
    def dcm2nii(self, input_directory, output_directory):
        self.inter_dcm2niix.run_dcm2niix(input_directory, output_directory)

    def get_nii_file(self, directory):
        return nib.load(directory)