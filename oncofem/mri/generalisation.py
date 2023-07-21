"""
Generalises input files

Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""

import os
import subprocess
from oncofem.helper.structure import Measure
from oncofem.interfaces.dcm2niix import Dcm2niix
from oncofem.interfaces.brainmage import BrainMaGe
from oncofem.helper.constant import GENERALISATION_PATH, DER_DIR, SRI24_T1, SRI24_T2, CAPTK_DIR
from oncofem.helper.general import get_path_file_extension, mkdir_if_not_exist
import ants
from fsl.utils.image.resample import resample
from fsl.data.image import Image
import nibabel as nib

class Generalisation:

    def __init__(self, mri):
        self.mri = mri
        self.dir = mri.study_dir + DER_DIR + mri.state.subject + os.sep + str(mri.state.date) + os.sep + GENERALISATION_PATH
        self.study_dir = self.mri.study_dir
        self.d2n = Dcm2niix()
        self.clean_outputs = True
        self.brain_mage = BrainMaGe()
        self.brain_mage.dev = "cpu"
        self.gen_shape = (240, 240, 155)
        mkdir_if_not_exist(self.dir)

    def dcm2niigz(self, measure: Measure):
        """
        converts input dcm file (folder) into packed nifti file

        # Arguments:
            measure: Measure contains all neccesary data
        # Returns:
            new niigz filepath 
        """
        dcm_dir = measure.dir_src
        niigz_dir = self.study_dir + DER_DIR + measure.subject + os.sep + str(measure.date) + os.sep + GENERALISATION_PATH
        self.d2n.f = measure.modality
        measure.dir_ngz = self.d2n.run_dcm2niix(dcm_dir, niigz_dir)
        measure.dir_act = measure.dir_ngz

    def bias_correction(self, measure: Measure):
        """
        Bias correction of the images
        """
        input_image = measure.dir_act
        measure.dir_bia = measure.dir_ngz.replace('.nii', '_bc.nii')
        measure.dir_act = measure.dir_bia
        image = ants.image_read(input_image)
        image_n4 = ants.n4_bias_field_correction(image)
        ants.image_write(image_n4, measure.dir_bia)

    def coregister_modality2atlas(self):
        """
        Co-registers different modalities into the same space. This should be done into a general atlas space
        """
        modalities = {"t1": "-t1", "t1ce": "-t1c", "t2": "-t2", "flair": "-fl"}
        self.mri.isFullModality()
        if self.mri.full_ana_modality:
            command = [CAPTK_DIR]
            command.append("BraTSPipeline.cwl")
            for measure in self.mri.state.measures:
                input_path = measure.dir_act
                measure.dir_cor = self.dir + str(measure.modality) + "_to_sri.nii.gz"
                measure.dir_act = measure.dir_cor
                command.append(modalities[measure.modality])
                command.append(input_path)

            command.append("-o")
            command.append(self.dir)
            command.append("-s")
            command.append("0")
            command.append("-b")
            command.append("0")
            p = subprocess.Popen(command, stdout=subprocess.PIPE)
            print(p.communicate())

        else:
            for measure in self.mri.state.measures:
                input_dir = measure.dir_bia
                path, file, file_wo_extension = get_path_file_extension(input_dir)
                file_sri24 = file_wo_extension + "_to_SRI.nii.gz"
                measure.dir_act = self.dir + file_sri24
                command = [CAPTK_DIR]
                command.append("Preprocessing.cwl")
                command.append("-i")
                command.append(input_dir)
                command.append("-rFI")
                if measure.modality == "t2":
                    command.append(SRI24_T2)
                else:
                    command.append(SRI24_T1)
                command.append("-o")
                command.append(self.dir + file_sri24)
                command.append("-reg")
                command.append("RIGID")
                p = subprocess.Popen(command, stdout=subprocess.PIPE)
                print(p.communicate())

    def skull_strip(self):
        """
        Skull strips the given input images
        """
        self.mri.isFullModality()
        if self.mri.full_ana_modality:
            input_files = [self.mri.t1_dir, self.mri.t2_dir, self.mri.t1ce_dir, self.mri.flair_dir]
            output_dir = self.dir + os.sep 
            self.brain_mage.multi_4_run(input_files, output_dir)

        else:
            for measure in self.mri.state.measures:
                path, file, file_wo_extension = get_path_file_extension(measure.dir_act)
                measure.dir_sks = self.dir + file_wo_extension + "_sks.nii.gz"
                measure.dir_brainmask = self.dir + file_wo_extension + "_brain.nii.gz"
                self.brain_mage.single_run(measure.dir_act, measure.dir_sks, measure.dir_brainmask)
                measure.dir_act = measure.dir_brainmask

    def resample2standard(self, file_dir: str):
        """
        Resamples given image into a standard shape. This is defined in config.ini
        """
        path, file, file_wo_extension = get_path_file_extension(file_dir)
        resample_dir = self.dir + os.sep + file_wo_extension + "_res.nii.gz"
        image = Image(file_dir)
        resample_image, resample_affine = resample(image, self.gen_shape)
        nifti_image = nib.Nifti1Image(resample_image, resample_affine)
        nib.save(nifti_image, resample_dir)
        return resample_dir

    def run_all(self):
        """
        Runs gen process:
            1. dcm2niigz
            2. Bias Correction (N4)
            3. Co-register axial, sagittal, coronal into one image (not implemented)
            4. Co-register into Atlas Space
            5. Skull strip 
            6. Resample onto Standard sample size
        """
        print("Begin generalisation")
        print("Full anatomical model: ", str(self.mri.isFullModality()))

        # 1 + 2 dcm2niigz + bias correction
        print("Begin dcm2niigz + bias correction")
        for measure in self.mri.state.measures:
            self.dcm2niigz(measure)
            self.bias_correction(measure)

        # 3 Co-register into atlas space
        print("Begin coregister 2 atlas")
        self.coregister_modality2atlas()

        # 4 Skull strip
        print("Begin skull strip")
        self.skull_strip()

        # 7. Actualize Paths
        for measure in self.mri.state.measures:
            if "bc_to_SRI_brain" in measure.dir_act:
                if "t1" in measure.dir_act:
                    self.mri.t1_dir = measure.dir_act
                elif "t2" in measure.dir_act:
                    self.mri.t2_dir = measure.dir_act
                elif "t1ce" in measure.dir_act:
                    self.mri.t1ce_dir = measure.dir_act
                elif "flair" in measure.dir_act:
                    self.mri.flair_dir = measure.dir_act