"""
Quick start tutorial

In this tutorial a training data set of the BraTS2020 challenge serves as simple test case for a simulation of
patient-specific data. The data consists of a standard magnetic resonance image and a respective segmentation of the
tumor compartments by an expert. In this code the basic steps of creating a numerical simulation of a patient-specific
test case are summarized. In order to perform this simplified test case, a simple two-phase model in the framework of
the Theory of Porous Media is extended about a concentration equation that represents the edema with resolved mobile
cancer cells. Therefore the governing equations read

 0 = div T - hatrhoF w_F
 0 = (nS)'_S + nS div x'_S - hatnS
 0 = div x'_S + div(nF w_F) - (hatrhoS / rhoSR + hatrhoF / rhoFR)
 0 = nF (cFt_m)'_S + div(nF cFt_m w_Ft) + cFt_m (div x'_S - hatrhoS / rhoS) - hatrhoFt / MFt_m

and are solved for the primary variables of the solid displacement u_S, the solid volume fraction nS, the fluid pressure
pF and the tumor cell concentration cFt. It is assumed, that the initial concentration is maximal at the solid tumour
segmentation and minimal at the outer edge of the edema. The spreading and growing of that area is then simulated. Since
no displacements are triggered in this first example and the pressure at the boundaries is set to zero and no pressure
gradients will evolve, the problem can be simplified into a poisson equation with

 0 = nF (cFt_m)'_S + div(nF cFt_m w_Ft) - hatrhoFt / MFt_m .

 Herein, the velocity of the mobile cancer cells reduce to its diffusive part

 nF cFt_m w_Ft = - DFt / (R Theta) grad cFt_m

with the diffusion parameter DFt, that becomes a scalar value for isotropic materials, the real gas constant R and the
temperature Theta. In this test case, the diffusion parameter varies for different microstructures (white-, grey matter
and cerebrospinal fluid) and the example shows the expected spreading of the mobile cancer cells into the preferred
growth directions.
"""
# Imports
import oncofem as of
of.
from oncofem.helper.fem_aux import BoundingBox, mark_facet
import dolfin as df
########################################################################################################################
# INPUT
#
# In a first step an input needs to be defined. To do so, first a study object is created. This study then creates a
# workspace on the hard drive with two subfolders 'der' and 'sol'. Herein, all derived  pre-processed results and final
# solutions are saved. The parent studies folder need to be set in the config.ini file. To compare the results of
# different subjects in a next hierarchical level a 'subject' needs to be created. This subject than can have several
# states of measurements taken at different time points. By means of that the initial state is created and the taken
# measurement files can be defined. A measurement can be created by the path to the relative file or directory and its
# modality. This ensures, that the information is interpreted correctly.
study = of.Study("test_nii2stl")
subj_1 = study.create_subject("Subject_1")
state_1 = subj_1.create_state("init_state")
path = of.ONCOFEM_DIR + "/data/tutorial/BraTS/BraTS20_Training_001/BraTS20_Training_001_"
measure_1 = state_1.create_measure(path + "t1.nii.gz", "t1")
measure_2 = state_1.create_measure(path + "seg.nii.gz", "seg")
########################################################################################################################
# MRI PRE-PROCESSING
#
# The mri entity can be set up by giving the particular state. All measurements of that state will be load and the most
# important structural MRIs (t1, t1gd, t2, flair, seg) are set. Furthermore, the affine of the first image is safed as a
# general quantity in the mri entity. First thing that is need to be evaluated, is the tumors spatial distribution and
# composition. Therefore, the tumor segmentation is set up. In this test case, this is already given via the input, this
# result can be set directly. To identify the particular compartment of the tumor (active, necrotic, edema), the command
# 'set_compartment_masks()' is executed.
mri = of.simulation.mri.MRI(state=state_1)
#mri.set_tumor_segmentation()
#mri.tumor_segmentation.seg_file = measure_2.dir_act
#mri.tumor_segmentation.set_compartment_masks()
# Since the brain consists of different areas with varying microstructural compositions and material properties, their
# spatial distributions is of interest. To get this information the so called 'white matter segmentation' is
# initialised. Herein, the default approach is to separate in between the white and grey matter, and the cerebrospinal
# fluid. For this task basically the fast algorithm of the software package fsl is used. Since, this algorithm works by
# separating the grey values of the image into three different compartments, the tumor will lead to failures in the
# resulting compartments. To overcome that issue the user can chose in between two different approaches. In this
# example, the 'tumor_entity_weighted' approach is chosen, which is based on the already known spatial composition of
# the tumor. Both approaches are discussed in tutorial 'tut_05_mri_structure_segmentation'. The user can further chose
# which inputs shall be used and the segmentation can be run with the 'run()' command. Since this step can take several
# time, the already performed output files from the already done calculation can be used from the data folder.
# Keep in mind, that the rest of the tutorial is build up on the tumor entity weighted approach, several adjustments
# need to be done for bias corrected approach.
mri.wm_mask = of.ONCOFEM_DIR + "/data/tutorial/tut_01/wm.nii.gz"
mri.gm_mask = of.ONCOFEM_DIR + "/data/tutorial/tut_01/gm.nii.gz"
mri.csf_mask = of.ONCOFEM_DIR + "/data/tutorial/tut_01/csf.nii.gz"
tumor_class_0 = of.ONCOFEM_DIR + "/data/tutorial/tut_01/tumor_class_pve_0.nii.gz"
tumor_class_1 = of.ONCOFEM_DIR + "/data/tutorial/tut_01/tumor_class_pve_1.nii.gz"
tumor_class_2 = of.ONCOFEM_DIR + "/data/tutorial/tut_01/tumor_class_pve_2.nii.gz"
input_tumor = [tumor_class_0, tumor_class_1, tumor_class_2]
########################################################################################################################
# SIMULATION
#
# With this now, all necessary information are gathered. This information needs to be translated into the quantities of
# the used model. Therefore, first a problem is set up, that holds all information for a numerical simulation. To
# interpret the gathered information with respect to the used model a field map generator is initiated. This entity
# generates approximate field quantities of the performed discontinuous distributions provided by the segmentations.
p = of.simulation.Problem(mri)
fmap = of.simulation.FieldMapGenerator(mri)
# Before a field can be mapped, of course first the domain is needed, where any field can be mapped on. This is done
# with the following code lines. Again the user can chose between different adjustments. A deeper look will be given in
# tutorial 'tut_06_field_map_generator'. Since this part again can be very time consuming the user can chose to perform
# this step, or take the files given in the data folder.
fmap.volume_resolution = 40
fmap.prim_mri_mod = p.mri.t1_dir
fmap.generate_geometry_file(p.mri.t1_dir)
#of.helper.io.nii2stl(fmap.prim_mri_mod, "nii2stl.stl",fmap.fmap_dir)
