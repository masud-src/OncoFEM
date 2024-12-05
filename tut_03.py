"""
Run automatic BraTS Mode

So far we have seen, that it is possible to import medical image data and run simulations from that input. In this
tutorial, a convenient function is presented which generates the necessary input data from a BraTS-like segmentations.
In the preserved segmentation file, following classes are present:

- healthy: 5
- edema: 2
- active: 4
- necrotic: 1

The 'run_BraTS()' method takes either a BraTS-like segmentation file or pre-generated mask files of the tumour and
results in mapped distributions of the tumour compartments (edema, active, necrotic) and in distributions for the
structural compartments (white and gray matter, csf). Note, that hereby the following assumptions are followed:

- edema has a maximum value at the area of solid tumour and goes to its minimum at the outer rim
- active goes from maximum at necrotic border to minimum at edema border
- necrotic is constant in necrotic area (this is assumed due to numerical issues)

In order to map quantities onto a mesh first thing that needs to be done is the creation of the mesh, like in the last
tutorial. Sometimes the mesh will not be generated properly and the user needs to refine and smooth. Both
functionalities are taken from mri2fem (https://github.com/kent-and/mri2fem). Therefore, the mesh generation is done
with subdivided commands. In this tutorial, the already mentioned methods of interpolating the segmentations, that
usually consist of discontinuous distributed classes into smooth functions are used in order to perform on BraTS-like
data with a convenient one line command.  The user can adjust the used settings with the following attributes:

fmap.interpolation_method = "linear"        - Interpolation method of scipy's griddata algorithm (linear, nearest)
fmap.structure_mapping_method = "const_wm"  - Approach of assumption for the tumor area (const_wm, mean_averaged_value)
fmap.volume_resolution = 16                 - Defines the volume resolution of the resulting finite element mesh
fmap.edema_max_value = 2.0                  - Maximum interpolation value of edema, solid tumor is assumed as plateau
fmap.edema_min_value = 1.0                  - Minimum interpolation value of edema, solid tumor is assumed as plateau
fmap.active_max_value = 2.0                 - Maximum interpolation value of active solid tumor
fmap.active_min_value = 1.0                 - Minimum interpolation value of active solid tumor
fmap.necrotic_max_value = 2.0               - Maximum interpolation value of active necrotic core
fmap.necrotic_min_value = 1.0               - Minimum interpolation value of active necrotic core

When the tutorial is finished, the user can continue analogously to previous tutorials.
"""
import oncofem as of
import os
########################################################################################################################
# INPUT
#
# A new study is set up and this time 'mask2.nii.gz' is load which contains 4 classes (healthy, edema, active, necrotic)
# that are load with the 'Measure()' command. The results of a pre-performed structure segmentation show the segmented
# distributions are summarised in the dictionary 'struc_seg'.
#
study = of.structure.Study("tut_03")
data_dir = os.getcwd() + os.sep + "data" + os.sep
measure_1 = of.structure.Measure(data_dir + "mask2.nii.gz", "mask")
wm_mask = data_dir + "wm.nii.gz"
gm_mask = data_dir + "gm.nii.gz"
csf_mask = data_dir + "csf.nii.gz"
struc_seg = {"wm": wm_mask, "gm": gm_mask, "csf": csf_mask}
########################################################################################################################
# INITIALISATION
#
# Apart from the field map generator entity, paths need to be defined for results and interim results.
#
fmap = of.FieldMapGenerator(study.der_dir)
out_dir = of.utils.io.set_out_dir(fmap.work_dir, of.field_map_generator.FIELD_MAP_PATH)
_, _, name = of.utils.io.get_path_file_extension(measure_1.dir_act)
stl_file = out_dir + name + ".stl"
stl_remesh_file = out_dir + name + "_remesh.stl"
stl_smooth_file = out_dir + name + "_smooth.stl"
mesh_file = out_dir + name + ".mesh"
#
# MESH GENERATION
#
# The first step is to generate a surface file (.stl) from the given NIFTI image.
#
of.utils.io.nii2stl(measure_1.dir_act, stl_file, out_dir)
#
# In this tutorial, the plain stl file is not sophisticating and the surface shall be remeshed and smoothed.
#
of.utils.io.remesh_surface(stl_file, stl_remesh_file, 1.0, 3)
of.utils.io.smoothen_surface(stl_remesh_file, stl_smooth_file, 1, 1.0)
#
# When the surface mesh is appropriate, a volume mesh (.mesh) can be created that is transformed into the xdmf format,
# that can be load into the field map generator entity.
volume_resolution = 10
of.utils.io.stl2mesh(stl_smooth_file, mesh_file, volume_resolution)
fmap.xdmf_file = of.utils.io.mesh2xdmf(mesh_file, out_dir)
fmap.dolfin_mesh = fmap.load_mesh(fmap.xdmf_file)
#
# MAPPING
#
# In the next lines the default parameters are assigned. Herein, the min and max values represent the borders of the
# interpolation and the respective method is set (linear, nearest). More information about the method can be found in
# the documentation of scipy.interpolate.griddata. Two different mappings need to be done. First is the distribution of
# the tumour compartments and secom
#
fmap.edema_min_value = 1.0
fmap.edema_max_value = 2.0
fmap.active_min_value = 1.0
fmap.active_max_value = 2.0
fmap.necrotic_min_value = 1.0
fmap.necrotic_max_value = 2.0
fmap.interpolation_method = "nearest"
fmap.structure_mapping_method = "const_wm"
fmap.set_struc_class_maps(struc_seg)
fmap.set_affine(measure_1.dir_act)
fmap.run_brats(brats_seg=measure_1.dir_act)
#
tumor_class_0 = data_dir + "tumor_class_pve_0.nii.gz"
tumor_class_1 = data_dir + "tumor_class_pve_1.nii.gz"
tumor_class_2 = data_dir + "tumor_class_pve_2.nii.gz"
input_tumor = [tumor_class_0, tumor_class_1, tumor_class_2]

fmap.work_dir = study.sol_dir
fmap.structure_mapping_method = "mean_averaged_value"
fmap.set_struc_class_maps(struc_seg)
fmap.set_affine(measure_1.dir_act)
fmap.run_brats(brats_seg=input_tumor)
