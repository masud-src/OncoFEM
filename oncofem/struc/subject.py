"""
# **************************************************************************#
#                                                                           #
# === Subject ==============================================================#
#                                                                           #
# **************************************************************************#
# Definition of subject class
#
# Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
#
# --------------------------------------------------------------------------#
"""

import datetime
import os
from .state import State

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
        state = State(ident, date)
        state.subject = self.ident
        state.date = date
        state.dir = str(date) + os.sep
        state.study_dir = self.study_dir
        self.states.append(state)
        return state
