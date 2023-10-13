"""
Process model set-ups, that describe the bio-chemical behaviour of cells and call cohorts can be defined in this package.
To ensure a certain standard to preserve a flexible usability all models shall be imported with their constructor. See 
for example the VerhulstKinetic. 

modules:
    process_model:      Definition of bio-chemical model base class. All implemented micro models shall be derived from
                        this parent class.
    verhulst_kinetic:   Definition of simple Verhulst-like kinetic, with a swich to turn on a solid - concentration 
                        coupling 

Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""
from .verhulst_kinetic import VerhulstKinetic
