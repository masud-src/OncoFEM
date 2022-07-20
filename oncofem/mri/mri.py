"""
# **************************************************************************#
#                                                                           #
# === MRI ==================================================================#
#                                                                           #
# **************************************************************************#
# Handling of medical images
#
# Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
#
# --------------------------------------------------------------------------#
"""



# **************************************************************************#
#      Classes                                                              #
# **************************************************************************#
# Definition of Classes

class white_matter_segmentation:
    def __init__(self):
        pass

class MRI:
    """
    t.b.d.
    """

    def __init__(self, study: Study):
        self.study = study
        self.generalizer = None
        self.list_full_modality = ["T1", "T1CE", "T2", "FL"]
        self.wm_seg = white_matter_segmentation()


    def set_up_generalisation(self):
        """
        Initializes generealisation module, now several options can be handled.
        # Arguments:
        """
        # initialize generalizer module
        self.generalizer = Generalisation(self.study)

    def run_generalisation(self, state: State):
        """
        Runs generalisation process:
            1. dcm2niigz
            2. Bias Correction (N4)
            3. Co-register axial, sagittal, coronal into one image (not implemented)
            4. Co-register into Atlas Space
            5. Skull strip 
            6. Resample onto Standard sample size
            7. DTI corrections of distortions (not implemented)
            8. PERFUSION corrections of distortions (not implemented)
        """
        print("Begin generalisation")

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

    def set_up_tumor_seg(self):
        """
        Initializes generealisation module, now several options can be handled.
        # Arguments:
        """
        # initialize generalizer module
        self.generalizer = Generalisation(self.study)
        pass

    def run_tumor_seg(self, state: State):
        pass

    def set_up_wm_seg(self, tumour_seg_file, t1_file):
        """
        t.b.d
        """
        self.tumour_segmentation_file = tumour_seg_file
        self.t1_file = t1_file

    def run_wm_seg(self):
        command = "fslmaths "
        command += self.tumour_segmentation_file
        command += "-div "
        command += self.tumour_segmentation_file
        command += "-mul 1e10"
        command += "seg_seg.nii.gz"
        run_shell_command(command)

        command = "fslmaths "
        command += self.t1_file
        command += "-sub "
        command += self.tumour_segmentation_file
        command += "-mul 1e10"
        command += "seg_seg.nii.gz"
        run_shell_command(command)

        # fslmaths BraTS20_Training_001_t1.nii.gz -sub seg_seg.nii.gz t1_seg.nii.gz
        # fslmaths t1_seg.nii.gz -thr -10000 t1-tumour.nii.gz
        #/usr/local/fsl/bin/fast -t 1 -n 3 -H 0.1 -I 10 -l 20.0 -B -b -o new t1-tumour.nii.gz  

        # fslmaths BraTS20_Training_003_seg.nii.gz -div BraTS20_Training_003_seg.nii.gz -mul 1e10 seg_seg.nii.gz
        # fslmaths BraTS20_Training_003_t1.nii.gz -sub seg_seg.nii.gz t1_seg.nii.gz
        # fslmaths t1_seg.nii.gz -thr -10000 t1-tumour.nii.gz
        # /usr/local/fsl/bin/fast -t 1 -n 3 -H 0.1 -I 10 -l 20.0 -B -b -o new t1-tumour.nii.gz  
        # EDEMA ?

    def set_up_DTI(self):
        pass

    def run_DTI(self, state):
        pass

    def set_up_perfusion(self):
        pass

    def run_perfusion(self, state):
        pass
