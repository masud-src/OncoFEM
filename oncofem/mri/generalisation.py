"""
# **************************************************************************#
#                                                                           #
# === Generalisation =======================================================#
#                                                                           #
# **************************************************************************#
# Generalises input files
#
# Co-author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
#
# --------------------------------------------------------------------------#
"""

from os import sep, listdir, path as ospath, rename
import subprocess
import csv

from oncofem.struct.study import Study
from nipype.interfaces.ants.segmentation import N4BiasFieldCorrection

class Generalisation:

    def __init__(self, study: Study):
        self.study = study
        self.d2n = Dcm2niix()
        self.n4 = N4BiasFieldCorrection()
        self.n4.inputs.dimension = 3
        self.n4.inputs.bspline_fitting_distance = 300
        self.n4.inputs.shrink_factor = 3
        self.n4.inputs.n_iterations = [50, 50, 30, 20]
        self.clean_outputs = True
        self.device = "cpu"
        
    def set_up_generalisation(self):
        """
        Initializes generealisation module, now several options can be handled.
        # Arguments:
        """
        # initialize generalizer module
        self.generalizer = Generalisation(self.study)

    def run_generalisation(self, state: State):
        """
        Runs gen process:
            1. dcm2niigz
            2. Bias Correction (N4)
            3. Co-register axial, sagittal, coronal into one image (not implemented)
            4. Co-register into Atlas Space
            5. Skull strip 
            6. Resample onto Standard sample size
            7. DTI corrections of distortions (not implemented)
            8. PERFUSION corrections of distortions (not implemented)
        """
        print("Begin gen")

        list_available_modality = [measure.modality for measure in state.measures]
        state.full_ana_modality = all(item in list_available_modality for item in self.list_full_modality)
        print("Full anatomical model: ", str(state.full_ana_modality))

        # 1 + 2 dcm2niigz + bias correction
        print("Begin dcm2niigz + bias correction")
        for measure in state.measures:
            self.generalizer.dcm2niigz(measure)
            self.generalizer.bias_correction(measure)

        # 3 Co-register and merge axial, sagittal, coronal into one image
        print("Begin coregister anatomical planes")
        self.generalizer.coregister_anatomical_planes()

        # 4 Co-register into atlas space
        print("Begin coregister 2 atlas")
        self.generalizer.coregister_modality2atlas(state)

        # 5 Skull strip
        print("Begin skull strip")
        self.generalizer.skull_strip(state)

        # 6. Clean files
        print("Begin clean files")


    def dcm2niigz(self, measure: Measure):
        """
        converts input dcm file (folder) into packed nifti file
        # Arguments:
            measure: Measure contains all neccesary data
        # Returns:
            new niigz filepath 
        """
        dcm_dir = measure.dir_src
        niigz_dir = self.study.der_dir + measure.subject + sep + str(measure.date) + sep + const.GENERALISATION_PATH
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
        command = ["/home/marlon/Software/CaPTk/1.8.1/captk"]

        if state.full_ana_modality:
            for measure in state.measures:
                if measure.modality == "T1":
                    input_path_T1 = measure.dir_bia
                    path, file, file_wo_extension = get_path_file_extension(input_path_T1)
                    measure.dir_cor = path + sep + "T1_to_SRI.nii.gz"
                    measure.dir_act = measure.dir_cor
                    state.t1_dir = measure.dir_cor
                if measure.modality == "T1CE":
                    input_path_T1CE = measure.dir_bia
                    path, file, file_wo_extension = get_path_file_extension(input_path_T1CE)
                    measure.dir_cor = path + sep + "T1CE_to_SRI.nii.gz"
                    measure.dir_act = measure.dir_cor
                    state.t1CE_dir = measure.dir_cor
                if measure.modality == "T2":
                    input_path_T2 = measure.dir_bia
                    path, file, file_wo_extension = get_path_file_extension(input_path_T2)
                    measure.dir_cor = path + sep + "T2_to_SRI.nii.gz"
                    measure.dir_act = measure.dir_cor
                    state.t2_dir = measure.dir_cor
                if measure.modality == "FL":
                    input_path_FLAIR = measure.dir_bia
                    path, file, file_wo_extension = get_path_file_extension(input_path_FLAIR)
                    measure.dir_cor = path + sep + "FL_to_SRI.nii.gz"
                    measure.dir_act = measure.dir_cor
                    state.flair_dir = measure.dir_cor

            command.append("BraTSPipeline.cwl")

            command.append("-t1")
            command.append(input_path_T1)
            command.append("-t1c")
            command.append(input_path_T1CE)
            command.append("-t2")
            command.append(input_path_T2)
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

            #for filename in listdir(path):
            #    if filename.startswith("T1CE") or filename.startswith("FL_"):
            #        f = ospath.join(path, filename)
            #        g = f
                    #g = g.replace("CE", "Gd")
                    #g = g.replace("FL", "FLAIR")
            #        rename(f, g)

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
                    command.append(const.PATH_SRI24_T2)
                else:
                    command.append(const.PATH_SRI24_T1)
                command.append("-o")
                command.append(path + sep + file_sri24)
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
            input_files = [state.t1_dir, state.t2_dir, state.t1CE_dir, state.flair_dir]
            output_dir = state.study_dir + const.DER_DIR + str(state.subject) + sep + str(state.dir) + const.GENERALISATION_PATH + "brain.nii.gz"
            brain_mage.multi_4_run(input_files, output_dir)

        else:
            for measure in state.measures:
                brain_mage.single_run(measure.dir_act, measure.dir_sks, measure.dir_brainmask)


    def resample2standard(self, image):
        """
        Resamples given image into a standard shape. This is defined in config.ini
        """
        pass

    def correct_distortions(self):
        """
        Corrects diffusion-weighted images and preserve them for generating bvectors and bvalues
        """
        pass
