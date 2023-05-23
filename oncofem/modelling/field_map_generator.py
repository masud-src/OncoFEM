"""
# **************************************************************************#
#                                                                           #
# === field map generator package  =========================================#
#                                                                           #
# **************************************************************************#
# In this sub-package of oncofem the field map generator is implemented.
# Herein, the following sub-modules can be found:
# 
#   field_map_generator - main module
#           
#       This module can be used as controller module for sub-declared tasks.
#       The main functionalies are: 
#           - the generation of xdmf files from nifti inputs,
#           - marking areas of the surface for boundary conditions,
#           - map the tumor compartments onto the generated mesh,
#           - map the heterogeneous distribution of white and gray matter
#             and csf onto the generated mesh, and
#           - map arbitrary fields onto the generated mesh.
# 
#   geometry
#       
#       The geometry module holds the Geometry class. Herein, the geometry
#       of a problem can be defined and elementary information about it is
#       gathered and can be collected in that entity. Furthermore, simple
#       academic examples can be created with predefined functions.
#
#
# Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
#
# --------------------------------------------------------------------------#
"""
import os

import scipy
from statsmodels.genmod.families.links import probit

import oncofem.helper.general
from oncofem.struc.problem import Problem
import oncofem.helper.constant as const
import oncofem.helper.io
import nibabel.loadsave
import dolfin
import numpy as np
from skimage.measure import regionprops
import skimage
import fsl
import nibabel as nib


class FieldMapGenerator:
    def __init__(self, problem: Problem):
        self.study_dir = problem.mri.study_dir
        self.mri = problem.mri
        state_dir = self.mri.study_dir + const.DER_DIR + self.mri.state.subject + os.sep + self.mri.state.dir
        self.fmap_dir = state_dir + const.FIELD_MAP_PATH 
        oncofem.helper.general.mkdir_if_not_exist(self.fmap_dir)
        self.prim_mri_mod = None
        self.mixed_wm_mask = None
        self.mixed_gm_mask = None
        self.mixed_csf_mask = None
        self.stl_file = self.fmap_dir + "geometry.stl"
        self.mesh_file = self.fmap_dir + "geometry.mesh"
        self.xdmf_file = None
        self.surf_xdmf_file = None
        self.dolfin_mesh = None
        self.volume_resolution = 16
        self.mapped_ede_file = None
        self.mapped_act_file = None
        self.mapped_nec_file = None
        self.mapped_wm_file = None
        self.mapped_gm_file = None
        self.mapped_csf_file = None
        self.interpolation_method = "linear"
        self.wms_mapping_method = "constant_wm"
        self.edema_max_value = 2.0
        self.edema_min_value = 1.0
        self.active_max_value = 2.0
        self.active_min_value = 1.0
        self.necrotic_max_value = 2.0
        self.necrotic_min_value = 1.0

    def generate_geometry_file(self, primary_mri_mod: str):
        """
        t.b.d.
        """
        self.prim_mri_mod = primary_mri_mod  
        # first nii2stl
        oncofem.io.nii2stl(self.prim_mri_mod, self.stl_file, 0, self.fmap_dir)
        # second stl2mesh
        oncofem.io.stl2mesh(self.stl_file, self.mesh_file, self.volume_resolution)
        # third msh2xmdf
        self.xdmf_file = oncofem.io.mesh2xdmf(self.mesh_file, self.fmap_dir)
        # load mesh
        self.dolfin_mesh = oncofem.io.load_mesh(self.xdmf_file)

    def mark_facet(self, bounding_boxes: list):
        """
        t.b.d.
        """
        mf_domain = dolfin.MeshFunction("size_t", self.dolfin_mesh, self.dolfin_mesh.topology().dim(),0)
        mf_facet = dolfin.MeshFunction("size_t", self.dolfin_mesh, self.dolfin_mesh.topology().dim()-1)
        for i, bounding_box in enumerate(bounding_boxes):
            bounding_box.mark(mf_facet, i+1)

        self.surf_xdmf_file = self.fmap_dir + "surface.xdmf"
        dolfin.XDMFFile.write(dolfin.XDMFFile(self.surf_xdmf_file), mf_facet)
        return mf_domain, mf_facet

    def interpolate_segm(self, image, name, plateau=None, hole=None, min_value=1.0, max_value=2.0, rest_value=0.0, method="linear"):
        if plateau is None:
            closed_vol = image
            center = skimage.measure.regionprops(image.astype(int))[0].centroid
            coords_max = (np.array([int(center[0])]), np.array([int(center[1])]), np.array([int(center[2])]))
        else:
            closed_vol = image + plateau
            coords_max = np.where(plateau == 1)
        if hole is not None:
            max_bound = skimage.segmentation.find_boundaries(hole.astype(int), mode="inner")
            coords_max = np.where(max_bound == 1)

        domain_shape = image.shape
        min_bound = skimage.segmentation.find_boundaries(closed_vol.astype(int), mode="outer")
        coords_min = np.where(min_bound == 1)

        max_mask = np.zeros(domain_shape, dtype=bool)
        for i in range(len(coords_max[0])):
            max_mask[(coords_max[0][i], coords_max[0][i], coords_max[0][i])] = True

        # Generate coordinates for the occupied and unoccupied points
        coords_interp = np.where(~(min_bound | max_mask))
        coords_inverse = np.where(closed_vol == 0)
        if hole is not None:
            coords_inverse += np.where(hole == 1)

        # Create arrays of coordinates for the occupied points
        coords_occupied_min = np.array(coords_min).T
        coords_occupied_max = np.array(coords_max).T

        # Create an array of values for the occupied points
        values_occupied_min = np.full(coords_occupied_min.shape[0], min_value)
        values_occupied_max = np.full(coords_occupied_max.shape[0], max_value)
        mask = np.in1d(values_occupied_min, values_occupied_max).reshape(values_occupied_min.shape[0], -1).all(axis=1)
        values_occupied_min = values_occupied_min[~mask]

        # Perform the interpolation
        print("start interpolation")
        interp_values = scipy.interpolate.griddata(np.concatenate((coords_occupied_min, coords_occupied_max)),
                                                   np.concatenate((values_occupied_min, values_occupied_max)),
                                                   coords_interp, method=method, fill_value=0.0)
        print("finished interpolation")
        # Create a 3D array with the minimum value
        values = np.full(domain_shape, 0.0)
        # Update the values array with the interpolated values
        coords_outside = (coords_inverse[0], coords_inverse[1], coords_inverse[2])
        values[coords_interp] = interp_values
        values[coords_max] = max_value
        values[coords_outside] = rest_value

        file_output = self.fmap_dir + name
        oncofem.io.write_field2nii(values, file_output, self.mri.affine)
        return file_output + ".nii.gz"

    def map_field(self, field_file, outfile, mesh_file=None):
        """
        t.b.d.
        """
        if mesh_file is None:
            mesh_file = self.xdmf_file
        image = nibabel.load(field_file)
        data = image.get_fdata()

        mesh = dolfin.Mesh()
        with dolfin.XDMFFile(mesh_file) as file:
            file.read(mesh)

        n = mesh.topology().dim()
        regions = dolfin.MeshFunction("double", mesh, n, 0)

        for cell in dolfin.cells(mesh):
            c = cell.index()

            # Convert to voxel space
            ijk = cell.midpoint()[:]

            # Round off to nearest integers to find voxel indices
            i, j, k = np.rint(ijk).astype("int")

            # Insert image data into the mesh function:
            regions.array()[c] = float(data[i, j, k])

        # Store regions in XDMF
        xdmf = dolfin.XDMFFile(mesh.mpi_comm(), outfile + ".xdmf")
        xdmf.parameters["flush_output"] = True
        xdmf.parameters["functions_share_mesh"] = True
        xdmf.write(mesh)
        xdmf.write(regions)
        xdmf.close()
        return outfile + ".xdmf"

    def run_edema_mapping(self):
        ede_ip = self.interpolate_segm(self.mri.ede_mask, "edema_ip", plateau=self.mri.act_mask + self.mri.nec_mask,
                                       min_value=self.edema_min_value, max_value=self.edema_max_value,
                                       method=self.interpolation_method)
        self.mapped_ede_file = self.map_field(ede_ip, self.fmap_dir + "edema")

    def run_solid_tumor_mapping(self):
        # Needed to change edema with necrotic...somehow lead to overwriting of edema
        # generate separated nii maps
        #nec_ip = self.interpolate_segm(self.mri.nec_mask, "necrotic_ip", min_value=self.necrotic_min_value,
        #                               max_value=self.necrotic_max_value, method=self.interpolation_method)
        act_ip = self.interpolate_segm(self.mri.act_mask, "active_ip", hole=self.mri.nec_mask, 
                                       min_value=self.active_min_value, max_value=self.active_max_value,
                                       method=self.interpolation_method)

        # hotfix, necrotic image has not nicely convex hull
        max_bound = skimage.segmentation.find_boundaries((self.mri.nec_mask + self.mri.act_mask).astype(int), mode="inner")
        solid_tumor = nib.Nifti1Image(self.mri.nec_mask + self.mri.act_mask - max_bound, self.mri.affine)
        active_tumor = nib.load(act_ip)
        nec = fsl.wrappers.fslmaths(active_tumor).div(active_tumor).mul(-1).add(solid_tumor).run()
        nib.save(nec, self.fmap_dir + "necrotic_ip.nii.gz")

        self.mapped_act_file = self.map_field(act_ip, self.fmap_dir + "active")
        self.mapped_nec_file = self.map_field(self.fmap_dir + "necrotic_ip.nii.gz", self.fmap_dir + "necrotic")

    def set_mixed_masks(self, classes=None):
        """
        Sets tumor classes analogous to the white and gray matter and csf. Needed for mean averaged value. List
        should have three entities. First for white matter, second for gray matter, third for csf.
        """
        if self.wms_mapping_method == "const_wm":
            tumor_mask = nib.Nifti1Image(self.mri.act_mask + self.mri.nec_mask + self.mri.ede_mask, self.mri.affine)
            fsl.wrappers.fslmaths(self.mri.wm_mask).add(tumor_mask).run(self.fmap_dir + "wm.nii.gz")
            self.mixed_wm_mask = self.fmap_dir + "wm.nii.gz"
            self.mixed_gm_mask = self.mri.gm_mask
            self.mixed_csf_mask = self.mri.csf_mask

        elif self.wms_mapping_method == "mean_averaged_value":
            fsl.wrappers.fslmaths(self.mri.wm_mask).add(classes[0]).run(self.fmap_dir + "wm.nii.gz")
            fsl.wrappers.fslmaths(self.mri.gm_mask).add(classes[1]).run(self.fmap_dir + "gm.nii.gz")
            fsl.wrappers.fslmaths(self.mri.csf_mask).add(classes[2]).run(self.fmap_dir + "csf.nii.gz")
            self.mixed_wm_mask = self.fmap_dir + "wm.nii.gz"
            self.mixed_gm_mask = self.fmap_dir + "gm.nii.gz"
            self.mixed_csf_mask = self.fmap_dir + "csf.nii.gz"

        elif self.wms_mapping_method == "tumor_entity_weighted":
            print("not implemented")
            pass

    def run_wm_mapping(self):
        """
        t.b.d.
        """
        self.mapped_wm_file = self.map_field(self.mixed_wm_mask, self.fmap_dir + "white_matter")
        self.mapped_gm_file = self.map_field(self.mixed_gm_mask, self.fmap_dir + "gray_matter")
        self.mapped_csf_file = self.map_field(self.mixed_csf_mask, self.fmap_dir + "csf")
