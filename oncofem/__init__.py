"""
Main module for OncoFEM.

Herein all sub packages are loaded for a simple usage of OncoFEM.
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
