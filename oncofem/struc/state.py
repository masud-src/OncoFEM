"""
Definition of state file

Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""

import datetime
import os
from . import measure

class State:
    """
    A state is an actual or reference state of a subject. A state can contain several measurements at a certain 
    time point. 

    *Methods*:
        create_measure: creates a measure that is directly bind to the state 
    """

    def __init__(self, ident: str, date: datetime.date):
        self.id = ident
        self.study_dir = None
        self.dir = str(date) + os.sep
        self.date = date
        self.subject = None
        self.full_ana_modality = None
        self.measures = []

    def create_measure(self, path: str, modality: str):
        m = measure.Measure(path, modality)
        m.date = self.date
        m.state = self.id
        m.subject = self.subject
        self.measures.append(m)
        return m
