"""
The handling of magenic resonance imaging series is coordinated within this mri class.

Class:
    MRI:    The base class for the pre-processing of the patient-specific input data. Main access point for all other
            mri related processes. All information is hold within this object. 
"""
from typing import Any, Union
import oncofem as of
import nibabel as nib
import fsl
import copy
import numpy as np

class MRI:
    """
    MRI is the base class for the pre-processing of the patient-specific input data. Herein, the measures of an input
    state are sorted and the basic structural modalities are available via the respective attribute. In order to 
    homogenize and further pre-process more attributes about image properties and masks of tumor and brain tissue 
    compartments are hold. Each sub-module for a particular task is bind via its respective attribute.

    *Attributes*:
        study_dir: String of study dir, is set by initializing with the particular state
        state: Respective input state
        t1_dir: Direction of t1 modality
        t1ce_dir: Direction of t1ce modality
        t2_dir: Direction of t2 modality
        flair_dir: Direction of flair modality
        seg_dir: Direction of segmentation
        full_ana_modality: Bool, check if all structural modalities are given (t1, t1ce, t2, flair)
        affine: Array of image affine, each modality is co-registered to that
        shape: Shape of the image each modality is co-registered to
        ede_mask: Binary mask image of the edema
        act_mask: Binary mask image of the active core
        nec_mask: Binary mask image of the necrotic core
        wm_mask: Binary mask image of the white matter
        gm_mask: Binary mask image of the gray matter
        csf_mask: Binary mask image of the cerebro-spinal fluid
        generalisation: holds the respective generalisation sub-module
        tumor_segmentation: holds the respective tumor segmentation sub-module
        wm_segmentation: holds the respective white matter segmentation sub-module

    *Methods*:
        set_generalisation: initializes the generalisation sub-module
        set_tumor_segmentation: initializes the tumor segmentation sub-module
        set_wm_segmentation: initializes the white matter segmentation sub-module
        set_affine: loads first given measurement and takes affine and shape
        load_measures: fills the respective arguments of the structural images and the segmentation
        isFullModality: checks if input state has full structural modality
        image2array: gives numpy array of image data
        image2mask: creates a mask of a given input image
        cut_area_from_image: cuts an area from an image
    """
    def __init__(self, state:of.State=None):
        self.t1_dir = None
        self.t1ce_dir = None
        self.t2_dir = None
        self.flair_dir = None
        self.seg_dir = None
        self.full_ana_modality = None
        self.affine = None
        self.shape = None
        self.ede_mask = None
        self.act_mask = None
        self.nec_mask = None
        self.wm_mask = None
        self.gm_mask = None
        self.csf_mask = None
        self.generalisation = None
        self.tumor_segmentation = None
        self.wm_segmentation = None
        if state is None:
            self.state = None
            self.study_dir = None
        else:
            self.state = state
            self.study_dir = state.study_dir
            self.load_measures()
            self.isFullModality()
            self.set_affine()

    def set_generalisation(self) -> None:
        """
        Sets generalisation entity and activates it. Access via self.generalisation. 
        """
        self.generalisation = of.mri.generalisation.Generalisation(self)

    def set_tumor_segmentation(self) -> None:
        """
        Sets tumor segmentation entity and activates it. Access via self.tumor_segmentation.
        """
        self.tumor_segmentation = of.mri.tumor_segmentation.TumorSegmentation(self)

    def set_wm_segmentation(self) -> None:
        """
        Sets white matter segmentation entity and activates it. Access via self.white_matter_segmentation.
        """
        self.wm_segmentation = of.mri.white_matter_segmentation.WhiteMatterSegmentation(self)

    def set_affine(self, image:nib.Nifti1Image=None) -> None:
        """
        Sets affine and shape of first measure of included state. The optional argument takes an nibabel Nifti1Image
        and takes the first measurement of the hold state of the mri entity if no argument is given. Affine and shape
        can be accessed via self.affine and self.shape.
        
        *Arguments*:
            image:      Optional nib.Nifti1Image, Default is self.state.measures[0].dir_act
        """
        if image is None:
            image = nib.load(self.state.measures[0].dir_act)
        self.affine = image.affine
        self.shape = image.shape

    def load_measures(self, state:of.State=None) -> None:
        """
        Loads the actual measure files and directs them to their correct modality within the mri entity. 
        
        *Arguments*:
            state:      Optional input state, if no argument is given, self.state.measures is taken
        """
        if state is None:
            state = self.state
        for measure in state.measures:
            if measure.modality == "t1":
                self.t1_dir = measure.dir_act
            elif measure.modality == "t1ce":
                self.t1ce_dir = measure.dir_act
            elif measure.modality == "t2":
                self.t2_dir = measure.dir_act
            elif measure.modality == "flair":
                self.flair_dir = measure.dir_act
            elif measure.modality == "seg":
                self.seg_dir = measure.dir_act

    def isFullModality(self, state:of.State=None) -> bool:
        """
        Checks if all structural gold standard entities are available. Returns boolean value.
        
        *Arguments*:
            state:      Optional input state, if no argument is given, self.state.measures is taken
        *Returns*:
            boolean value
        """
        if state is None:
            state = self.state
        list_available_modality = [measure.modality for measure in state.measures]
        list_full_modality = ["t1", "t1ce", "t2", "flair"]
        self.full_ana_modality = all(item in list_available_modality for item in list_full_modality)
        return self.full_ana_modality

    @staticmethod
    def image2array(image_dir:str) -> tuple[Any, Any, Any]:
        """
        Takes a directory of an image and gives a numpy array.
        
        *Arguments*:
            image_dir:      String of a Nifti image directory
        *Returns*:
            numpy array of image data, shape, affine
        """
        orig_image = nib.load(image_dir)
        return copy.deepcopy(orig_image.get_fdata()), orig_image.shape, orig_image.affine

    @staticmethod
    def image2mask(image_dir:str, compartment:int=None, inner_compartments:list[int]=None) -> np.ndarray:
        """
        Gives deep copy of original image with selected compartments.
        
        *Arguments*:
            image_dir:          String to Nifti image
            compartment:        Int, identifier of compartment that shall be filtered
            inner_compartments: List of inner compartments that also are included in the mask
        *Returns*:
            mask:               Numpy array of the binary mask
        """
        mask, _, _ = MRI.image2array(image_dir)
        unique = list(np.unique(mask))
        unique.remove(compartment)
        for outer in unique:
            mask[np.isclose(mask, outer)] = 0.0
        mask[np.isclose(mask, compartment)] = 1.0
        if inner_compartments is not None:
            for comp in inner_compartments:
                mask[np.isclose(mask, comp)] = 1.0
                unique.remove(comp)
        return mask

    @staticmethod
    def cut_area_from_image(input_image:str, area_mask:nib.Nifti1Image, inverse:bool=False) -> Union[None, nib.Nifti1Image]:
        """
        Cuts an area of that image.
        
        *Arguments*:
            input_image:    String of path to Nifti image
            area_mask:      Mask array
            inverse         Bool, true for inverse cut
        
        *Returns*:
            optionally returns the image or writes it next to the input image
        """
        if inverse:
            area_mask = fsl.wrappers.fslmaths(area_mask).mul(-1).add(1).run()

        return fsl.wrappers.fslmaths(input_image).mul(area_mask).run()
