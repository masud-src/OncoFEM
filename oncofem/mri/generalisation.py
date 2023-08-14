"""
Generalisation of the input files for further processing of the segmentations.

Classes:
    Generalisation:     Holds all information and is the entry point for every process done. Each process can be
                        perfomed solitary or everything is done in a clustered run command.
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
    """
    The generalisation entity is the entry point of patient-specific magnetic resonance image series. Herein, the 
    images are set into a comparable scope for further investigations.
    
    *Arguments*:
        mri:            Base class is hold for directory information
        dir:            String of generalisation outputs
        study_dir:      String of study direction
        d2n:            dcm2niix entity is hold which converts DICOM images into Nifti images
        brain_mage:     BrainMaGe entity which performs the skull stripping
        gen_shape:      Tuple of integers that represents the general shape of an 3D mri scan.
        
    *Methods*:
        dcm2niigz:                  Processes a dicom image series into Nifti files and packs it
        bias_correction:            Performs the bias correction to set every image on the same level of intensity
        coregister_modality2atlas:  Coregisters the images into the direction of the SRI24 atlas
        skull_strip:                Performs the skull stripping with BrainMaGe
        resample2standard:          Resamples the images to a standard resolution
        run_all:                    Runs all commands in a clustered command
    """

    def __init__(self, mri):
        self.mri = mri
        self.gen_dir = mri.work_dir + GENERALISATION_PATH
        self.d2n = Dcm2niix()
        self.brain_mage = BrainMaGe()
        self.brain_mage.dev = "cpu"
        self.gen_shape = (240, 240, 155)

    def dcm2niigz(self, measure:Measure) -> None:
        """
        converts input dcm file (folder) into packed nifti file

        # Arguments:
            measure: Measure contains all neccesary data
        """
        mkdir_if_not_exist(self.gen_dir)
        dcm_dir = measure.dir_src
        niigz_dir = self.gen_dir
        self.d2n.f = measure.modality
        measure.dir_ngz = self.d2n.run_dcm2niix(dcm_dir, niigz_dir)
        measure.dir_act = measure.dir_ngz

    def bias_correction(self, measure:Measure) -> None:
        """
        Bias correction of the images
        
        *Arguments*:
            measure: Measure contains all necessary data
        """
        mkdir_if_not_exist(self.gen_dir)
        input_image = measure.dir_act
        measure.dir_bia = measure.dir_ngz.replace('.nii', '_bc.nii')
        measure.dir_act = measure.dir_bia
        image = ants.image_read(input_image)
        image_n4 = ants.n4_bias_field_correction(image)
        ants.image_write(image_n4, measure.dir_bia)

    def coregister_modality2atlas(self) -> None:
        """
        Co-registers different modalities into the same space. This should be done into a general atlas space
        """
        mkdir_if_not_exist(self.gen_dir)
        modalities = {"t1": "-t1", "t1ce": "-t1c", "t2": "-t2", "flair": "-fl"}
        self.mri.isFullModality()
        if self.mri.full_ana_modality:
            command = [CAPTK_DIR]
            command.append("BraTSPipeline.cwl")
            for measure in self.mri.state.measures:
                input_path = measure.dir_act
                measure.dir_cor = self.gen_dir + str(measure.modality) + "_to_sri.nii.gz"
                measure.dir_act = measure.dir_cor
                command.append(modalities[measure.modality])
                command.append(input_path)

            command.append("-o")
            command.append(self.gen_dir)
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
                measure.dir_act = self.gen_dir + file_sri24
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
                command.append(self.gen_dir + file_sri24)
                command.append("-reg")
                command.append("RIGID")
                p = subprocess.Popen(command, stdout=subprocess.PIPE)
                print(p.communicate())

    def skull_strip(self) -> None:
        """
        Skull strips the given input images
        """
        mkdir_if_not_exist(self.gen_dir)
        self.mri.isFullModality()
        if self.mri.full_ana_modality:
            input_files = [self.mri.t1_dir, self.mri.t2_dir, self.mri.t1ce_dir, self.mri.flair_dir]
            output_dir = self.gen_dir + os.sep 
            self.brain_mage.multi_4_run(input_files, output_dir)

        else:
            for measure in self.mri.state.measures:
                path, file, file_wo_extension = get_path_file_extension(measure.dir_act)
                measure.dir_sks = self.gen_dir + file_wo_extension + "_sks.nii.gz"
                measure.dir_brainmask = self.gen_dir + file_wo_extension + "_brain.nii.gz"
                self.brain_mage.single_run(measure.dir_act, measure.dir_sks, measure.dir_brainmask)
                measure.dir_act = measure.dir_brainmask

    def resample2standard(self, file_dir:str) -> str:
        """
        Resamples given image into a standard shape.
        
        *Arguments*:
            file_dir:       String of file path
        *Returns*:
            resample_dir:   String of path of resampled image
        
        """
        mkdir_if_not_exist(self.gen_dir)
        path, file, file_wo_extension = get_path_file_extension(file_dir)
        resample_dir = self.gen_dir + os.sep + file_wo_extension + "_res.nii.gz"
        image = Image(file_dir)
        resample_image, resample_affine = resample(image, self.gen_shape)
        nifti_image = nib.Nifti1Image(resample_image, resample_affine)
        nib.save(nifti_image, resample_dir)
        return resample_dir

    def run_all(self) -> None:
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

        print("Begin dcm2niigz + bias correction")
        for measure in self.mri.state.measures:
            self.dcm2niigz(measure)
            self.bias_correction(measure)

        print("Begin coregister 2 atlas")
        self.coregister_modality2atlas()

        print("Begin skull strip")
        self.skull_strip()

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
