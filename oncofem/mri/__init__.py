"""
In this package the mri pre-processor of OncoFEM is implemented. To gather all information a base structure class mri
is defined in its respective module. Herein, all image related informations are collected and this object remains as a
control file for all image related processes. Therefore, all submodules are hold and can be run from that object. The 
sub-modules split into their different task. In general first a generalisation is needed. In the tumor segmentation, 
the composition and spatial distribution of the particular tumor compartments are extracted. The white matter 
segmentation uses fast from fsl to separate the brain microstructures into their respective classes (white and gray
matter and cerebrospinal fluid).

Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""
from .mri import MRI
from . import generalisation
from .tumor_segmentation import tumor_segmentation
from . import white_matter_segmentation
