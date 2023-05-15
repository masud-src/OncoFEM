"""
White matter segmentation module

Author: Marlon Suditsch
"""
import fsl.wrappers.fslmaths
import fsl.wrappers.fast
import nibabel as nib
import numpy as np

import oncofem.mri.mri
from oncofem.struc.state import State
from oncofem.helper import general as gen
from oncofem.helper.general import mkdir_if_not_exist

class WhiteMatterSegmentation:
    """
    White matter segmentation main class
    Attributes
        fast: Fast Parameters, herein the parameters for the fsl interface are hold

    Methods
        set_input_wm_seg: 
    """
    def __init__(self, mri):
        self.mri = mri
        self.study_dir = mri.study_dir
        self.work_dir = None
        self.brain_dirs = None
        self.tumor_dirs = None
        self.input_files_dir = None
        self.tumor_seg_dir = None
        self.tumor_handling_approach = "mean_averaged_value"
        self.tumor_handling_classes = 3
        self.n_b_const = 3
        self.n_input = None
        self.input_type = None
        self.output_basename = None

    def set_input_wm_seg(self, input_files_dir: list, tumor_seg_dir, work_dir=None, modality=None):
        """
        Set input of white matter segmentation

        Arguments
            input_files_dir: List, takes all structural input files gathered in a list
            tumor_seg_nii: tumor segmentation file. Corresponds to tumor segmentation file from tumor segmentation
            modality: OPTIONAL, when only one input file is given, this should identify the given file for better result
        """
        if work_dir is not None:
            self.work_dir = work_dir
        mkdir_if_not_exist(work_dir)
        self.n_input = len(input_files_dir)
        if self.n_input==1:
            valid_modality = {"t1", "t1ce", "t2", "flair"}
            if modality not in valid_modality:
                raise ValueError("results: status must be one of %r." % valid_modality)
            self.input_type = modality

        self.input_files_dir = input_files_dir
        self.tumor_seg_dir = tumor_seg_dir

    def run_single_segmentation(self, basename, files_list, n_classes):
        """
        runs fast segmentation algorithm in default with variable input files 
        """
        fsl.wrappers.fast(files_list, basename, n_classes)

    def run_all(self):
        """
        runs the white matter segmentation
        """
        image_tumor_mask = nib.Nifti1Image(self.mri.ede_mask + self.mri.act_mask + self.mri.nec_mask, self.mri.affine)
        for modality in self.input_files_dir:
            _, _, file = gen.get_path_file_extension(modality)
            for b in [True, False]:
                if b:
                    image = oncofem.mri.mri.cut_area_from_image(modality, image_tumor_mask, True)
                    nib.save(image, self.work_dir + file + "-woTumor.nii.gz")
                else:
                    image = oncofem.mri.mri.cut_area_from_image(modality, image_tumor_mask, False)
                    nib.save(image, self.work_dir + file + "-withTumor.nii.gz")

        brain_files = []
        tumor_files = []
        for modality in self.input_files_dir:
            _, _, file = gen.get_path_file_extension(modality)
            brain_files.append(self.work_dir + file + "-woTumor.nii.gz")

        self.run_single_segmentation(self.work_dir + "wms_Brain", brain_files, self.n_b_const)  # 2: white matter, 1: gray matter 0: CSF
        self.brain_dirs = [self.work_dir + "wms_Brain_pve_" + str(i) + "nii.gz" for i in range(self.n_b_const)]

        if self.tumor_handling_approach == "mean_averaged_value":
            # just returns classification based on intensity
            for modality in self.input_files_dir:
                _, _, file = gen.get_path_file_extension(modality)
                tumor_files.append(self.work_dir + file + "-withTumor.nii.gz")
            self.run_single_segmentation(self.work_dir + "wms_Tumor", tumor_files, self.tumor_handling_classes)
            self.tumor_dirs = [self.work_dir + "wms_Tumor_pve_" + str(i) + "nii.gz" for i in range(self.tumor_handling_classes)]

        if self.tumor_handling_approach == "tumor_entity_weighted":
            # get segmentation and separate in three classes, cut class-wise from mri and normalise 
            image_ede_mask = nib.Nifti1Image(self.mri.ede_mask, self.mri.affine)
            image_act_mask = nib.Nifti1Image(self.mri.act_mask, self.mri.affine)
            image_nec_mask = nib.Nifti1Image(self.mri.nec_mask, self.mri.affine)
            masks = {"edema_distr": image_ede_mask, "active_distr": image_act_mask, "necrotic_distr": image_nec_mask}

            for modality in self.input_files_dir:
                _, _, file = gen.get_path_file_extension(modality)
                for key in masks:
                    image = oncofem.mri.mri.cut_area_from_image(modality, masks[key], False)
                    array = image.get_fdata()
                    array[array == 0] = -1
                    array = (array - array[array > 0].min()) / (array.max() - array[array > 0].min())
                    array[array < 0] = -1
                    array = array + 1
                    image = nib.Nifti1Image(array, self.mri.affine)
                    nib.save(image, self.work_dir + str(key) + "-withTumor.nii.gz")
                    self.tumor_dirs.append(self.work_dir + str(key) + "-withTumor.nii.gz")

        if self.tumor_handling_approach == "mixed":
            pass
