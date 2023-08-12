"""
In models all structural elements of a convolutional neural network are clustered. Main code is from T. Henry and
colleagues. For understanding its expended about this documentation

modules:
    augmentation_blocks:    Performs random flip and rotation batch wise, and reverse it if needed.
    layers:                 With this the user can perform translations from dcm files to nifti.
    unet:                   With this the user can create meshes from nifti files.
"""
from .augmentation_blocks import DataAugmenter
from .layers import get_norm_layer
from .unet import Unet,  EquiUnet, Att_EquiUnet
