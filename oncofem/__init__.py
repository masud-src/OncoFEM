"""
This is the base entry point of OncoFEM. Herein all basic functionalities are
imported.


Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""
import oncofem.helper
import oncofem.interfaces
from .modelling.field_map_generator.geometry import Geometry #, create_2D_QuarterCircle, create_2D_QuarterCircle_Tumor
import oncofem.modelling.field_map_generator.geometry as geom
from .modelling.base_model import solver
from .struc import Study, Measure, State, Subject, Problem
from .helper import io, general, constant, auxillaries
from .interfaces import fsl, nii2mesh, dcm2niix, greedy, brainmage
from .mri.mri import MRI
from .modelling.field_map_generator.field_map_generator import FieldMapGenerator
from .modelling.base_model.stochastic_model import Stochastic_Model
from .modelling.bio_chem_models.bio_chem_models import BioChemModel
