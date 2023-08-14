"""
In this white matter segmentation module the heterogeneous distributions are identified.

Class:
    WhiteMatterSegmentation:    Basically uses fast of FSL for the segmentation. Herein, a mixture of Gaussian 
                                distributes the relevant classes of white and grey matter and cerebrospinal fluid in the
                                brain area.
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
    Basically uses fast of FSL for the segmentation. Herein, a mixture of Gaussian distributes the relevant classes of 
    white and grey matter and cerebrospinal fluid in the brain area.
    
    *Attributes*:
        mri:                        MRI entity in order to have all image related information including directories 
        dir:                        String of the directory to this entity which is created in its initialisation.
        input_files_dir:            List of the input images. Its recommended to only use one image.
        tumor_handling_approach:    String, for option switch. (bias corrected, tumor_entity_weighted)
        tumor_handling_classes:     Int, number of different tumour entities. Default is 3
        brain_handling_classes:     Int, number of different healthy brain entities. Default is 3
        brain_dirs:                 List of Strings of the generated brain files

    Methods
        set_input_wm_seg:               Sets the input files
        bias_corrected_approach:        Implements the subroutine for the bias corrected approach
        tumor_entity_weighted_approach: Implements the subroutine for the tumor entity weighted approach
        run:                            Runs the white matter segmentation
    """
    def __init__(self, mri):
        self.mri = mri
        self.wms_dir = mri.work_dir + const.WHITE_MATTER_SEGMENTATION_PATH
        self.input_files_dir = None
        self.tumor_handling_approach = "bias_corrected"
        self.tumor_handling_classes = 3
        self.brain_handling_classes = 3
        self.brain_dirs = None

    def set_input_wm_seg(self, input_files_dir:list) -> None:
        """
        Set input files of white matter segmentation.

        Arguments
            input_files_dir: List, takes all structural input files gathered in a list
            tumor_seg_nii: tumor segmentation file. Corresponds to tumor segmentation file from tumor segmentation
        """
        self.input_files_dir = input_files_dir

    @staticmethod
    def single_segmentation(basename:str, files_list:list[str], n_classes:int) -> None:
        """
        runs fast segmentation algorithm in default with variable input files.
        
        *Arguments*:
            basename:       String for base name of outputfiles
            files_list:     List of input images
            n_classes:      Number of segmentation classes
        """
        fsl.wrappers.fast(files_list, basename, n_classes)

    def bias_corrected_approach(self) -> None:
        """
        Sub-routine for the bias corrected approach, where fast is run again on the cut area of the tumor segmentation.
        """
        tumor_files = []
        # just returns classification based on intensity
        for modality in self.input_files_dir:
            _, _, file = gen.get_path_file_extension(modality)
            tumor_files.append(self.wms_dir + file + "-withTumor.nii.gz")
        self.single_segmentation(self.wms_dir + "tumor_class", tumor_files, self.tumor_handling_classes)

    def tumor_entity_weighted_approach(self) -> None:
        """
        Sub-routine for the tumor entity weighted approach, where the actual tumor segmentation is used to create masks
        for cuts of the original image. The original intensities are normalised, which gives three entities with 
        non-binary distribution from 1 to 2 with an offset that allows also 0.
        """
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
                nib.save(image, self.wms_dir + str(key) + ".nii.gz")

    def run(self) -> None:
        """
        Performs the white matter segmentation with the preset options and with the defined input parameters.
        """
        mkdir_if_not_exist(self.wms_dir)
        image_tumor_mask = nib.Nifti1Image(self.mri.ede_mask + self.mri.act_mask + self.mri.nec_mask, self.mri.affine)
        for modality in self.input_files_dir:
            _, _, file = gen.get_path_file_extension(modality)
            for b in [True, False]:
                if b:
                    image = oncofem.mri.mri.MRI.cut_area_from_image(modality, image_tumor_mask, True)
                    nib.save(image, self.wms_dir + file + "-woTumor.nii.gz")
                else:
                    image = oncofem.mri.mri.MRI.cut_area_from_image(modality, image_tumor_mask, False)
                    nib.save(image, self.wms_dir + file + "-withTumor.nii.gz")

        brain_files = []
        for modality in self.input_files_dir:
            _, _, file = gen.get_path_file_extension(modality)
            brain_files.append(self.wms_dir + file + "-woTumor.nii.gz")

        self.single_segmentation(self.wms_dir + "wms_Brain", brain_files, self.brain_handling_classes)  # 2: white matter, 1: gray matter 0: CSF
        if self.brain_handling_classes == 3:
            os.rename(self.wms_dir + "wms_Brain_pve_0.nii.gz", self.wms_dir + "csf.nii.gz")
            os.rename(self.wms_dir + "wms_Brain_pve_1.nii.gz", self.wms_dir + "gm.nii.gz")
            os.rename(self.wms_dir + "wms_Brain_pve_2.nii.gz", self.wms_dir + "wm.nii.gz")
            self.mri.wm_mask = self.wms_dir + "wm.nii.gz"
            self.mri.gm_mask = self.wms_dir + "gm.nii.gz"
            self.mri.csf_mask = self.wms_dir + "csf.nii.gz"
        else:
            self.brain_dirs = [self.wms_dir + "wms_Brain_pve_" + str(i) + "nii.gz" for i in range(self.brain_handling_classes)]

        if self.tumor_handling_approach == "bias_corrected":
            self.bias_corrected_approach()

        if self.tumor_handling_approach == "tumor_entity_weighted":
            self.tumor_entity_weighted_approach()
