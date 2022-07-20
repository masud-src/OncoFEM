# --- Project description ----------------------------------------------
# Controller File 
#
# ----------------------------------------------------------------------
# Imports
import datetime
import oncofem as of

###########################################################################################################
# Step 1: Define Input data
#
#

study = of.Study("Brain_Tumour")  # Create study object
subject_W1 = study.create_subject("W1")  # Create subject object
date = datetime.date(1996, 10, 27)
state_W1_t1 = subject_W1.create_state("t1", date)  # Create State
measure_W1T1 = state_W1_t1.create_measure("PATH_TO_FILE", "T1")
measure_W1T1CE = state_W1_t1.create_measure("PATH_TO_FILE", "T1CE")
measure_W1T2 = state_W1_t1.create_measure("PATH_TO_FILE", "T2")
measure_W1FL = state_W1_t1.create_measure("PATH_TO_FILE", "FL")

# Create study object
study = of.Study("Brain_Tumour") 

# Create subject object
subject_W1 = study.create_subject("W1")

# Create State
state_W1_t1 = subject_W1.create_state("t1", datetime.date(1996, 10, 27))

# Create T1 measurement from subject W1
measure_W1T1 = state_W1_t1.create_measure("/media/marlon/data/studies/Brain_Tumour/src/W1/DCM/1996_10_27/T1_AX_FSPGR_3D_PRE")
measure_W1T1.modality = "T1"

# Create T2 measurement from subject W1
measure_W1T1Gd = state_W1_t1.create_measure("/media/marlon/data/studies/Brain_Tumour/src/W1/DCM/1996_10_27/T1_AX_FSPGR_3D_POST")
measure_W1T1Gd.modality = "T1CE"

# Create T2 measurement from subject W1
measure_W1T2 = state_W1_t1.create_measure("/media/marlon/data/studies/Brain_Tumour/src/W1/DCM/1996_10_27/T2_AX_GRE")
measure_W1T2.modality = "T2"

# Create T2 measurement from subject W1
measure_W1FLAIR = state_W1_t1.create_measure("/media/marlon/data/studies/Brain_Tumour/src/W1/DCM/1996_10_27/T1_AX_FLAIR_POST")
measure_W1FLAIR.modality = "FL"

# Create subject object
#subject_W2 = study.create_subject("W2")

# Create State
#state_W2_t1 = subject_W2.create_state("t1", datetime.date(1996,11,1))

# Create T1 measurement from subject W2
#measure_W2T1 = state_W2_t1.create_measure("/media/marlon/data/studies/Brain_Tumour/src/W2/DCM/1996_11_01/T1_AX_FSPGR_3D_PRE")
#measure_W2T1.modality = "T1"

# Create T2 measurement from subject W2
#measure_W2T2 = state_W2_t1.create_measure("/media/marlon/data/studies/Brain_Tumour/src/W2/DCM/1996_11_01/T2_AX_FRFSEXL_2MM")
#measure_W2T2.modality = "T2"

###########################################################################################################
# Initialize MRI Entity 
#
mr_unit = of.mri.MRI(study)

for subject in study.subjects:
    print("Begin subject:  ", str(subject.ident))
    for state in subject.states:
        print("Begin state:  ", str(state.id))

###########################################################################################################
# Step 2: Generalisation
# In this step all subjects and modalities are processed to exclude brain
#

        mr_unit.set_up_generalisation()
        # possible modifications
        mr_unit.run_generalisation(state)

###########################################################################################################

###########################################################################################################
# Step 3: Tumor Segmentation
# In this step the tumor is segmented
#

        mr_unit.set_up_tumor_segmentation()
        # possible modifications
        mr_unit.run_tumor_segmentation(state)

###########################################################################################################

###########################################################################################################
# Step 4: White Matter Segmentation
# 
#

        mr_unit.set_up_white_matter_segmentation(state.tumor_segmentation, state.t1_dir)
        # possible modifications
        mr_unit.white_matter_segmentation.tumor_area_handle = 1  # 1: volume fractions 2: linear

        mr_unit.run_white_matter_segmentation()


###########################################################################################################

###########################################################################################################
# Step 5: DTI
# FDT, EDDY, TOPUP
#
        mr_unit.set_up_DTI()
        # possible modifications
        mr_unit.run_DTI(state)

###########################################################################################################

###########################################################################################################
# Step 6: Perfusion
# BASIL
#

        mr_unit.set_up_perfusion()
        # possible modifications
        mr_unit.run_perfusion(state)

###########################################################################################################

###########################################################################################################
# Step 7: Preparation for IBVPs
#
#



#study = Study("Brain_Tumour")
#subject = Subject("W1")
#measure = Measure("SRI/1996_10_25_CWRU_labels.nii.gz")
#subject.measures.append(measure)
#doc="/media/marlon/data/MRI_data/IvyGAP/data/Multi-Institutional_Paired_Expert_Segmentations_SRI_images-atlas-annotations/3_Annotations_SRI/CWRU/W1/W1_1996.10.25_CWRU_labels.nii.gz"
#t = TumorMapGenerator(study)
#t.read_labelprop_from_image(doc)
#t.generate_necrotic_tumor_map()
#t.generate_solid_tumor_map()
#t.generate_edema_map()

###########################################################################################################

###########################################################################################################
# Step 8: Set-up Processors
#
#
#
#

###########################################################################################################

###########################################################################################################
# Step 9: Set-up Problems
#
#
#
#

###########################################################################################################

###########################################################################################################
# Step 10: Process Problems
#
#
#
#

###########################################################################################################
