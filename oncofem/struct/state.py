"""
# **************************************************************************#
#                                                                           #
# === State ================================================================#
#                                                                           #
# **************************************************************************#
# Definition of state file
#
# Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
#
# --------------------------------------------------------------------------#
"""
import datetime
import oncofem

class State:
    """
    A state is an actual or reference state of a subject. A state can contain several measurements at a certain time point. 
    """

    def __init__(self, ident: str, date: datetime.date):
        self.id = ident
        self.study_dir = None
        self.t1_dir = None
        self.t1CE_dir = None
        self.t2_dir = None
        self.flair_dir = None
        self.dir = None
        self.date = date
        self.tumor_seg = None
        self.subject = None
        self.full_ana_modality = None
        self.measures = []

    def create_measure(self, path: str, modality: str):
        measure = oncofem.struct.Measure(path, modality)
        measure.date = self.date
        measure.state = self.id
        measure.subject = self.subject
        self.measures.append(measure)
        return measure
