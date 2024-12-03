"""
SIMULATION field map generator tutorial

In order to have a look on the field map generator, first a study needs to be set up with a test state of a subject. In
order to have all necessary objects ready the tumor and structure segmentation needs to be initialized. The field map
generator can than be initialized either, like in tutorial 'tut_01_quickstart', with an optional argument of an mri
object or without any argument. Keep in mind, that the field map generator object needs a mri object, wherein the tumor
and structure segmentation is done. Without this information there is no need to make use of the field mapping.
Nevertheless still its functionality of generating a geometry file could be useful. The user can adjust the preferences
of the object via the following arguments:

fmap.interpolation_method = "linear"        - Interpolation method of scipy's griddata algorithm (linear, nearest)
fmap.structure_mapping_method = "const_wm"  - Approach of assumption for the tumor area (const_wm, mean_averaged_value)
fmap.volume_resolution = 16                 - Defines the volume resolution of the resulting finite element mesh
fmap.edema_max_value = 2.0                  - Maximum interpolation value of edema, solid tumor is assumed as plateau
fmap.edema_min_value = 1.0                  - Minimum interpolation value of edema, solid tumor is assumed as plateau
fmap.active_max_value = 2.0                 - Maximum interpolation value of active solid tumor
fmap.active_min_value = 1.0                 - Minimum interpolation value of active solid tumor
fmap.necrotic_max_value = 2.0               - Maximum interpolation value of active necrotic core
fmap.necrotic_min_value = 1.0               - Minimum interpolation value of active necrotic core

The first step is in general the finite mesh generation which can be controlled with the volume resolution. The upper
level command generate_geometry_file clusters the commands:

of.helper.io.nii2stl(self.prim_mri_mod, self.stl_file, 0, self.out_dir)        # first nii2stl
of.helper.io.stl2mesh(self.stl_file, self.mesh_file, self.volume_resolution)    # second stl2mesh
of.helper.io.mesh2xdmf(self.mesh_file, self.out_dir)                           # third msh2xmdf
of.helper.io.load_mesh(self.xdmf_file)                                          # load mesh

If problems acure in this step, the user may want to improve the mesh generation with the following helper commands:

remesh_surface(stl_input:str, output:str, max_edge_length:float, n:int, do_not_move_boundary_edges:bool=False) -> None:

smoothen_surface(stl_input:str, output:str, n:int=1, eps:float=1.0, preserve_volume:bool=True) -> None:

where the first remeshes the surface of a stl surface mesh and the second smoothes the surface of a stl surface mesh.
Both are taken from mri2fem (https://github.com/kent-and/mri2fem).

The mapping of the segmentation can be performed afterwards. Basically two different commands are used:

fmap.interpolate(mask_file, name, plateau, min_value, max_value, method)

of.helper.io.map_field(interpolated_file, output, mesh:optional)

So, first the regarded quantity is interpolated into a 3 dimensional space. Herein, it is possible to set different
ranges and to define plateau areas, where the maximum value shall be assigned. The interpolation is done with scipy's
griddata algorithm, which offers two different approaches for 3D interpolations (linear, nearest). After this numerical
costly application the resulting field is mapped onto the actual calculation domain.

The high level commands

fmap.run_solid_tumor_mapping()
fmap.run_edema_mapping()

perform the named commands with pre-defined preferences. Since, the material parameters for the tumor area are hard to
distinguish, the user has the ability with the

fmap.set_mixed_masks()

command to follow two different approaches for the assumption of the material parameters in the tumor distributed area.
First (fmap.structure_mapping_method = "const_wm") assumes the tumor areas as constant white matter area. Second
(fmap.structure_mapping_method = "mean_averaged_value") assumes the first class of tumor area with white matter, second
class of tumor area with grey matter and last with cerebrospinal fluid. This assumption is done, according to the
similar color in the respective MRI scans. With

fmap.run_structure_mapping()

finally, the structural entities (white matter, grey matter and cerebrospinal fluid) are mapped onto the geometry.
"""
import oncofem as of
import os
########################################################################################################################
# INPUT
study = of.structure.Study("tut_03")
data_dir = os.getcwd() + os.sep + "data" + os.sep
measure_1 = of.structure.Measure(data_dir + "mask.nii.gz", "mask")
wm_mask = data_dir + "/data/tutorial/tut_01/wm.nii.gz"
gm_mask = data_dir + "/data/tutorial/tut_01/gm.nii.gz"
csf_mask = data_dir + "/data/tutorial/tut_01/csf.nii.gz"
struc_seg = [wm_mask, gm_mask, csf_mask]
tumor_class_0 = data_dir + "/data/tutorial/tut_01/tumor_class_pve_0.nii.gz"
tumor_class_1 = data_dir + "/data/tutorial/tut_01/tumor_class_pve_1.nii.gz"
tumor_class_2 = data_dir + "/data/tutorial/tut_01/tumor_class_pve_2.nii.gz"
input_tumor = [tumor_class_0, tumor_class_1, tumor_class_2]
########################################################################################################################
# FIELD MAP GENERATOR
fmap = of.FieldMapGenerator(study.der_dir)
# Generate
fmap.volume_resolution = 10
fmap.nii2dolfin_mesh(measure_1.dir_act)
# Set up tumour mapping
fmap.edema_min_value = 1.0
fmap.edema_max_value = 2.0
fmap.active_min_value = 1.0
fmap.active_max_value = 2.0
fmap.necrotic_min_value = 1.0
fmap.necrotic_max_value = 2.0
fmap.interpolation_method = "nearest"
fmap.run_brats(brats_seg=input_tumor, struc_seg=struc_seg)
