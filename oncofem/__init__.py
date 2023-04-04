"""
# **************************************************************************#
#                                                                           #
# === Main package  ========================================================#
#                                                                           #
# **************************************************************************#
# In this module an interface to the dcm2nii package is implemented.
# With this the user can perform translations from dcm files to nifti.
# 
#
# Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
#
# --------------------------------------------------------------------------#
"""
import oncofem.helper
import oncofem.interfaces
#import oncofem.struc.measure
#from .struc import Study, Measure, State, Subject, Problem
#from .helper import io, general, constant, auxillaries
#from .interfaces import fsl, nii2mesh, dcm2niix, greedy, brainmage
#from .mri.mri import MRI
#from .modelling.field_map_generator.field_map_generator import FieldMapGenerator
from .modelling.base_model.stochastic_model import Stochastic_Model
from .modelling.bio_chem_models.bio_chem_models import BioChemModel
