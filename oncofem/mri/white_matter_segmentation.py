"""
White matter segmentation module

Author: Marlon Suditsch
"""
import fsl.wrappers.fslmaths
import fsl.wrappers.fast

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
    def __init__(self, state: State):
        self.study_dir = state.study_dir
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

    def cut_tumor(self):
        """
        cut out tumor region and separates all modalities in a tumor and a non tumor region. 
        Therefore, in a first step a tumor mask and its inverse is created.
        """
        fsl.wrappers.fslmaths(self.tumor_seg_dir).div(self.tumor_seg_dir).run(self.work_dir + "tmask.nii.gz")
        fsl.wrappers.fslmaths(self.work_dir + "tmask.nii.gz").mul(-1).add(1).run(self.work_dir + "tmask_inverse.nii.gz")
        masks = ["tmask.nii.gz", "tmask_inverse.nii.gz"]
        for modality in self.input_files_dir:
            _, _, file = gen.get_path_file_extension(modality)
            for mask in masks:
                if mask == masks[0]:
                    fsl.wrappers.fslmaths(modality).mul(self.work_dir + mask).run(self.work_dir + file + "-withTumor.nii.gz")
                else:
                    fsl.wrappers.fslmaths(modality).mul(self.work_dir + mask).run(self.work_dir + file + "-woTumor.nii.gz")

    def run_single_segmentation(self, basename, files_list, n_classes):
        """
        runs fast segmentation algorithm in default with variable input files 
        """
        fsl.wrappers.fast(files_list, basename, n_classes)

    def run_all(self):
        """
        runs the white matter segmentation
        """
        self.cut_tumor()

        brain_files = []
        tumor_files = []
        for modality in self.input_files_dir:
            _, _, file = gen.get_path_file_extension(modality)
            brain_files.append(self.work_dir + file + "-woTumor.nii.gz")
            tumor_files.append(self.work_dir + file + "-withTumor.nii.gz")

        self.run_single_segmentation(self.work_dir + "wms_Brain", brain_files, self.n_b_const)  # 2: white matter, 1: gray matter 0: CSF
        self.brain_dirs = [self.work_dir + "wms_Brain_pve_" + str(i) + "nii.gz" for i in range(self.n_b_const)]
        
        if self.tumor_handling_approach == "mean_averaged_value":
            # just returns classification based on intensity
            self.run_single_segmentation(self.work_dir + "wms_Tumor", tumor_files, self.tumor_handling_classes)
            self.tumor_dirs = [self.work_dir + "wms_Tumor_pve_" + str(i) + "nii.gz" for i in range(self.tumor_handling_classes)]
        if self.tumor_handling_approach == "tumor_entity_weighted":
            # get segmentation and separate in three classes, cut class-wise from mri and normalise 
            
            self.work_dir = [self.work_dir + "wms_Brain_pve_" + i for i in range(self.n_b_const)]
        if self.tumor_handling_approach == "mixed":
            pass
