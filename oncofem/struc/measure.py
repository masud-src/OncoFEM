"""
Definition of general structure giving functionalities: Measure

Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""

class Measure:
    """
    A measure is the actual measure of a mri modality. It usually comes raw in dicom format. In order to pre-process
    there are particular arguments for the conversion into nifti format (dir_ngz), for bias correction (dir_bia), for
    co-registration (dir_cor) and the skull stripped version (dir_sks). 
    """
    def __init__(self, path: str, modality: str):
        self.dir_src = path
        self.dir_act = path
        self.dir_ngz = None
        self.dir_bia = None
        self.dir_cor = None
        self.dir_sks = None
        self.dir_brainmask = None
        self.date = None
        self.state = None
        self.subject = None
        self.modality = modality
