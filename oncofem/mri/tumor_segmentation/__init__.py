"""
In this sub-package of oncofem, the tumor segmentation based on T. Henry's convolutional neural network is implemented.
https://github.com/lescientifik/open_brats2020

Packages: 
    models                  Herein, the U-Net and its structural entities are implemented. Therefore, a module for the 
                            augmentation blocks, a module of the layers and the particular unet can be found. Files are
                            mostly original taken from the named github repository.

Modules:
    tumor_segmentation:     This is the implemented interface of the convolutional neural network made up by Henry. 
                            The inference and the training are combined in this file. A couple of changes have been 
                            necessary to adapt the code into OncoFEM as well as the functionality of different input 
                            images. Nevertheless, most of the code is the work of T. Henry and colleagues.
    utils:                  Herein, necessary functionalites are implemented.
"""
from . import tumor_segmentation