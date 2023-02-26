"""
# **************************************************************************#
#                                                                           #
# === Measure ==============================================================#
#                                                                           #
# **************************************************************************#
# Definition of general structure giving functionalities: Measure
#
# Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
#
# --------------------------------------------------------------------------#
"""

class Measure:
    """
    t.b.d.
    """
    def __init__(self, path: str, modality: str):
        self.dir_src = path
        self.dir_act = None
        self.dir_ngz = None
        self.dir_bia = None
        self.dir_cor = None
        self.dir_sks = None
        self.dir_brainmask = None
        self.date = None
        self.state = None
        self.subject = None
        self.modality = modality
        self.plane = None
        self.machine = None
        self.comments = None
