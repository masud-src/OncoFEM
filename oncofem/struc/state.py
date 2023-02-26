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
import os
import oncofem

class State:
    """
    A state is an actual or reference state of a subject. A state can contain several measurements at a certain time point. 
    """

    def __init__(self, ident: str, date: datetime.date):
        self.id = ident
        self.study_dir = None
        self.t1_dir = None
        self.t1ce_dir = None
        self.t2_dir = None
        self.flair_dir = None
        self.dir = str(date) + os.sep
        self.date = date
        self.tumor_seg = None
        self.subject = None
        self.full_ana_modality = None
        self.measures = []

    def create_measure(self, path: str, modality: str):
        measure = oncofem.struc.measure.Measure(path, modality)
        measure.date = self.date
        measure.state = self.id
        measure.subject = self.subject
        self.measures.append(measure)
        if modality == "t1":
            self.t1_dir = path
        if modality == "t1ce":
            self.t1ce_dir = path
        if modality == "t2":
            self.t2_dir = path
        if modality == "flair":
            self.flair_dir = path
        if modality == "seg":
            self.tumor_seg = path
        return measure

    def isFullModality(self):
        list_available_modality = [measure.modality for measure in self.measures]
        list_full_modality = ["t1", "t1ce", "t2", "fl"]
        self.full_ana_modality = all(item in list_available_modality for item in list_full_modality)
