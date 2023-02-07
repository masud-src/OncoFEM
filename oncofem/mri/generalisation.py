"""
# **************************************************************************#
#                                                                           #
# === Generalisation =======================================================#
#                                                                           #
# **************************************************************************#
# Generalises input files
#
# Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
#
# --------------------------------------------------------------------------#
"""

import os
import subprocess
from oncofem.struc.state import State
from oncofem.struc.measure import Measure
from oncofem.interfaces.dcm2niix import Dcm2niix
from oncofem.interfaces.brainmage import BrainMaGe
from oncofem.helper.constant import GENERALISATION_PATH, DER_DIR, PATH_SRI24_T1, PATH_SRI24_T2
from oncofem.helper.general import get_path_file_extension
from nipype.interfaces.ants.segmentation import N4BiasFieldCorrection

class Generalisation:

    def __init__(self, state: State):
        self.state = state
        self.study_dir = state.study_dir
        self.d2n = Dcm2niix()
        self.n4 = N4BiasFieldCorrection()
        self.n4.inputs.dimension = 3
        self.n4.inputs.bspline_fitting_distance = 300
        self.n4.inputs.shrink_factor = 3
        self.n4.inputs.n_iterations = [50, 50, 30, 20]
        self.clean_outputs = True
        self.device = "cpu"

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
        self.n4.inputs.input_image = measure.dir_act
        measure.dir_bia = measure.dir_ngz.replace('.nii', '_bc.nii')
        measure.dir_act = measure.dir_bia
        self.n4.inputs.output_image = measure.dir_bia
        self.n4.run()

    def coregister_anatomical_planes(self):
        """
        co-registers different anatomical images don't know if will work
        """
        pass

    def coregister_modality2atlas(self, state: State):
        """
        Co-registers different modalities into the same space. This should be done into a general atlas space
        """
        # TODO: check strings T1 to t1 etc
        command = ["/home/marlon/Software/CaPTk/1.8.1/captk"]
        command.append("BraTSPipeline.cwl")
        if state.full_ana_modality:
            for measure in state.measures:
                if measure.modality == "T1":
                    input_path_T1 = measure.dir_bia
                    path, file, file_wo_extension = get_path_file_extension(input_path_T1)
                    measure.dir_cor = path + os.sep + "T1_to_SRI.nii.gz"
                    measure.dir_act = measure.dir_cor
                    state.t1_dir = measure.dir_cor
                    command.append("-t1")
                    command.append(input_path_T1)
                if measure.modality == "T1CE":
                    input_path_T1CE = measure.dir_bia
                    path, file, file_wo_extension = get_path_file_extension(input_path_T1CE)
                    measure.dir_cor = path + os.sep + "T1CE_to_SRI.nii.gz"
                    measure.dir_act = measure.dir_cor
                    state.t1CE_dir = measure.dir_cor
                    command.append("-t1c")
                    command.append(input_path_T1CE)
                if measure.modality == "T2":
                    input_path_T2 = measure.dir_bia
                    path, file, file_wo_extension = get_path_file_extension(input_path_T2)
                    measure.dir_cor = path + os.sep + "T2_to_SRI.nii.gz"
                    measure.dir_act = measure.dir_cor
                    state.t2_dir = measure.dir_cor
                    command.append("-t2")
                    command.append(input_path_T2)
                if measure.modality == "FL":
                    input_path_FLAIR = measure.dir_bia
                    path, file, file_wo_extension = get_path_file_extension(input_path_FLAIR)
                    measure.dir_cor = path + os.sep + "FL_to_SRI.nii.gz"
                    measure.dir_act = measure.dir_cor
                    state.flair_dir = measure.dir_cor
                    command.append("-fl")
                    command.append(input_path_FLAIR)

            command.append("-o")
            path, file, file_wo_extension = get_path_file_extension(input_path_T1)
            command.append(path)
            command.append("-s")
            command.append("0")
            command.append("b")
            command.append("0")
            p = subprocess.Popen(command, stdout=subprocess.PIPE)
            print(p.communicate())

        else:
            for measure in state.measures:
                input_dir = measure.dir_bia
                path, file, file_wo_extension = get_path_file_extension(input_dir)
                file_sri24 = file_wo_extension + "_to_SRI.nii.gz"

                command.append("Preprocessing.cwl")
                command.append("-i")
                command.append(input_dir)
                command.append("-rFI")
                if measure.modality == "T2":
                    command.append(PATH_SRI24_T2)
                else:
                    command.append(PATH_SRI24_T1)
                command.append("-o")
                command.append(path + os.sep + file_sri24)
                command.append("-reg")
                command.append("RIGID")
                p = subprocess.Popen(command, stdout=subprocess.PIPE)
                print(p.communicate())

    def skull_strip(self, state: State):
        """
        Skull strips the given input images
        """
        brain_mage = BrainMaGe(state)
        brain_mage.dev = self.device

        if state.full_ana_modality:
            input_files = [state.t1_dir, state.t2_dir, state.t1ce_dir, state.flair_dir]
            output_dir = state.study_dir + DER_DIR + str(state.subject) + os.sep + str(state.dir) + GENERALISATION_PATH + "brain.nii.gz"
            brain_mage.multi_4_run(input_files, output_dir)

        else:
            for measure in state.measures:
                brain_mage.single_run(measure.dir_act, measure.dir_sks, measure.dir_brainmask)

    def resample2standard(self, image):
        """
        Resamples given image into a standard shape. This is defined in config.ini
        """
        pass

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
        print("Full anatomical model: ", str(self.state.full_ana_modality))

        # 1 + 2 dcm2niigz + bias correction
        print("Begin dcm2niigz + bias correction")
        for measure in self.state.measures:
            self.dcm2niigz(measure)
            self.bias_correction(measure)

        # 3 Co-register and merge axial, sagittal, coronal into one image
        print("Begin coregister anatomical planes")
        self.coregister_anatomical_planes()

        # 4 Co-register into atlas space
        print("Begin coregister 2 atlas")
        self.coregister_modality2atlas(self.state)

        # 5 Skull strip
        print("Begin skull strip")
        self.skull_strip(self.state)

        # 6. Clean files
        print("Begin clean files")
        self.resample2standard(self.state)

        # 7. Actualize Paths  # TODO: Fix paths to after all processing, so the paths in mri are actual
