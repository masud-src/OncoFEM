"""
# **************************************************************************#
#                                                                           #
# === Subject ==============================================================#
#                                                                           #
# **************************************************************************#
# Definition of subject file
#
# Co-author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
# Co-author: Maximilian Brodbeck <maximilian.brodbeck@isd.uni-stuttgart.de>
#
# --------------------------------------------------------------------------#
"""

import datetime
from os import sep
import oncofem

class Subject:
    """
    A Subject is a clinical specimen that is under investigation in some
    point. The subject usually provides patient-specific data. 
    """

    def __init__(self, ident: str):
        self.ident = ident
        self.study_dir = None
        self.states = []

    def create_state(self, ident: str, date: datetime.date):
        state = oncofem.struct.state.State(ident)
        state.subject = self.ident
        state.date = date
        state.dir = str(date) + sep
        state.study_dir = self.study_dir
        self.states.append(state)
        return state
