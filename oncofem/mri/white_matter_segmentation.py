"""
White matter segmentation module
"""
import fsl.wrappers.fslmaths
import fsl.wrappers.fast
import nibabel as nib
import os
import oncofem.mri.mri
from oncofem.helper import general as gen
from oncofem.helper.general import mkdir_if_not_exist
import oncofem.helper.constant as const

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
        self.dir = mri.study_dir + const.DER_DIR + mri.state.subject + os.sep + str(mri.state.date) + os.sep + const.WHITE_MATTER_SEGMENTATION_PATH
        mkdir_if_not_exist(self.dir)
        self.input_files_dir = None
        self.tumor_handling_approach = "mean_averaged_value"
        self.tumor_handling_classes = 3
        self.brain_handling_classes = 3
        self.output_basename = None
        self.brain_dirs = None

    def set_input_wm_seg(self, input_files_dir: list):
        """
        Set input of white matter segmentation

        Arguments
            input_files_dir: List, takes all structural input files gathered in a list
            tumor_seg_nii: tumor segmentation file. Corresponds to tumor segmentation file from tumor segmentation
        """
        self.input_files_dir = input_files_dir

    @staticmethod
    def single_segmentation(basename, files_list, n_classes):
        """
        runs fast segmentation algorithm in default with variable input files 
        """
        fsl.wrappers.fast(files_list, basename, n_classes)

    def bias_corrected_approach(self):
        tumor_files = []
        # just returns classification based on intensity
        for modality in self.input_files_dir:
            _, _, file = gen.get_path_file_extension(modality)
            tumor_files.append(self.dir + file + "-withTumor.nii.gz")
        self.single_segmentation(self.dir + "tumor_class", tumor_files, self.tumor_handling_classes)

    def tumor_entity_weighted_approach(self):
        # get segmentation and separate in three classes, cut class-wise from mri and normalise 
        image_ede_mask = nib.Nifti1Image(self.mri.ede_mask, self.mri.affine)
        image_act_mask = nib.Nifti1Image(self.mri.act_mask, self.mri.affine)
        image_nec_mask = nib.Nifti1Image(self.mri.nec_mask, self.mri.affine)
        masks = {"edema": image_ede_mask, "active": image_act_mask, "necrotic": image_nec_mask}

        for modality in self.input_files_dir:
            _, _, file = gen.get_path_file_extension(modality)
            for key in masks:
                image = oncofem.mri.mri.MRI.cut_area_from_image(modality, masks[key], False)
                array = image.get_fdata()
                array[array == 0] = -1
                array = (array - array[array > 0].min()) / (array.max() - array[array > 0].min())
                array[array < 0] = -1
                array = array + 1
                image = nib.Nifti1Image(array, self.mri.affine)
                nib.save(image, self.dir + str(key) + ".nii.gz")

    def run(self):
        """
        runs the white matter segmentation
        """
        image_tumor_mask = nib.Nifti1Image(self.mri.ede_mask + self.mri.act_mask + self.mri.nec_mask, self.mri.affine)
        for modality in self.input_files_dir:
            _, _, file = gen.get_path_file_extension(modality)
            for b in [True, False]:
                if b:
                    image = oncofem.mri.mri.MRI.cut_area_from_image(modality, image_tumor_mask, True)
                    nib.save(image, self.dir + file + "-woTumor.nii.gz")
                else:
                    image = oncofem.mri.mri.MRI.cut_area_from_image(modality, image_tumor_mask, False)
                    nib.save(image, self.dir + file + "-withTumor.nii.gz")

        brain_files = []
        for modality in self.input_files_dir:
            _, _, file = gen.get_path_file_extension(modality)
            brain_files.append(self.dir + file + "-woTumor.nii.gz")

        self.single_segmentation(self.dir + "wms_Brain", brain_files, self.brain_handling_classes)  # 2: white matter, 1: gray matter 0: CSF
        if self.brain_handling_classes == 3:
            os.rename(self.dir + "wms_Brain_pve_0.nii.gz", self.dir + "csf.nii.gz")
            os.rename(self.dir + "wms_Brain_pve_1.nii.gz", self.dir + "gm.nii.gz")
            os.rename(self.dir + "wms_Brain_pve_2.nii.gz", self.dir + "wm.nii.gz")
            self.mri.wm_mask = self.dir + "wm.nii.gz"
            self.mri.gm_mask = self.dir + "gm.nii.gz"
            self.mri.csf_mask = self.dir + "csf.nii.gz"
        else:
            self.brain_dirs = [self.dir + "wms_Brain_pve_" + str(i) + "nii.gz" for i in range(self.brain_handling_classes)]

        if self.tumor_handling_approach == "bias_corrected":
            self.bias_corrected_approach()

        if self.tumor_handling_approach == "tumor_entity_weighted":
            self.tumor_entity_weighted_approach()
