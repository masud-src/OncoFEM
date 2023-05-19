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

import oncofem.helper.general
from oncofem.struc.problem import Problem
import oncofem.helper.general as gen
import oncofem.helper.io
import nibabel.loadsave
import dolfin
import numpy as np
from skimage.measure import regionprops
import skimage

class BoundingBox(dolfin.SubDomain):
    """

    """
    def __init__(self, mesh, x_bounds=None, y_bounds=None, z_bounds=None):
        dolfin.SubDomain.__init__(self)
        self.x_bounds = x_bounds 
        self.y_bounds = y_bounds 
        self.z_bounds = z_bounds
        self.mesh = mesh

    def inside(self, x, on_boundary):
        if self.x_bounds is None:
            x_max = np.max(self.mesh.coordinates()[:, 0])
            x_min = np.min(self.mesh.coordinates()[:, 0])
            x_b = (x_min, x_max)
        else:
            x_b = self.x_bounds
        if self.y_bounds is None:
            y_max = np.max(self.mesh.coordinates()[:, 1])
            y_min = np.min(self.mesh.coordinates()[:, 1])
            y_b = (y_min, y_max)
        else:
            y_b = self.y_bounds
        if self.z_bounds is None:
            z_max = np.max(self.mesh.coordinates()[:, 2])
            z_min = np.min(self.mesh.coordinates()[:, 2])
            z_b = (z_min, z_max)
        else:
            z_b = self.z_bounds
        cond1 = dolfin.between(x[0], x_b)
        cond2 = dolfin.between(x[1], y_b)
        cond3 = dolfin.between(x[2], z_b)
        in_bounding_box = cond1 and cond2 and cond3
        return in_bounding_box and on_boundary

class MapAverageMaterialProperty(dolfin.UserExpression):
    def __init__(self, values, distributions, weights, **kwargs):
        self.distributions = distributions
        self.values = values
        self.weights = weights
        super().__init__(**kwargs)

    def eval_cell(self, values, x, cell):
        sum = 0
        for i in range(len(self.weights)):
            sum += self.values[i] * self.weights[i] * self.distributions[i][cell.index]
        values[0] = sum

class GeometryParam:
    def __init__(self):
        self.stl_file = None
        self.mesh_file = None
        self.xdmf_file = None
        self.surf_xdmf_file = None
        self.dolfin_mesh = None

class FieldMapGenerator:
    def __init__(self, problem: Problem):
        self.study_dir = problem.mri.study_dir
        self.mri = problem.mri
        self.fmap_dir = None
        self.wms_dir = None
        self.geom = GeometryParam()
        self.tumor_seg_file = None
        self.mapped_edema_file = None
        self.mapped_solid_tumor_file = None
        self.mapped_necrotic_file = None
        self.mapped_wm_file = None
        self.mapped_gm_file = None
        self.mapped_csf_file = None
        self.interpolation_method = "linear"
        self.edema_max_value = 2.0
        self.edema_min_value = 1.0
        self.active_max_value = 2.0
        self.active_min_value = 1.0
        self.necrotic_max_value = 2.0
        self.necrotic_min_value = 1.0

    def set_fmap_dir(self, dir: str):
        """
        sets directory for field mapping 
        """
        self.fmap_dir = oncofem.helper.general.mkdir_if_not_exist(dir)
        self.geom.stl_file = self.fmap_dir + "geometry.stl"
        self.geom.mesh_file = self.fmap_dir + "geometry.mesh"

    def set_primary_mri_mod(self, primary_mri_mod):
        self.prim_mri_mod = primary_mri_mod

    def mark_facet(self, bounding_boxes: list):
        mf_domain = dolfin.MeshFunction("size_t", self.geom.dolfin_mesh, self.geom.dolfin_mesh.topology().dim(),0)
        mf_facet = dolfin.MeshFunction("size_t", self.geom.dolfin_mesh, self.geom.dolfin_mesh.topology().dim()-1)
        for i, bounding_box in enumerate(bounding_boxes):
            bounding_box.mark(mf_facet, i+1)

        self.surf_xdmf_file = self.fmap_dir + "surface.xdmf"
        dolfin.XDMFFile.write(dolfin.XDMFFile(self.surf_xdmf_file), mf_facet)
        return mf_domain, mf_facet

    def meshfunction_2_function(self, mf: dolfin.MeshFunction, fs: dolfin.FunctionSpace):
        """maps meshfunction to functionspace. Only works with constant meshfunction space and linear functionspace"""
        v2d = dolfin.vertex_to_dof_map(fs)
        u = dolfin.Function(fs)
        u.vector()[v2d] = mf.array()
        return u

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
            mesh_file = self.geom.xdmf_file
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

    def generate_geometry_file(self):
        # first nii2stl
        oncofem.io.nii2stl(self.prim_mri_mod, self.geom.stl_file, 0, self.fmap_dir)
        # second stl2mesh
        oncofem.io.stl2mesh(self.geom.stl_file, self.geom.mesh_file)
        # third msh2xmdf
        self.geom.xdmf_file = oncofem.io.mesh2xdmf(self.geom.mesh_file, self.fmap_dir)
        # load mesh
        self.geom.dolfin_mesh = oncofem.io.load_mesh(self.geom.xdmf_file)

    def run_tumor_mapping(self):
        # Needed to change edema with necrotic...somehow lead to overwriting of edema
        # generate separated nii maps
        nec_ip = self.interpolate_segm(self.mri.nec_mask, "necrotic_ip", min_value=self.necrotic_min_value,
                                       max_value=self.necrotic_max_value, method=self.interpolation_method)

        act_ip = self.interpolate_segm(self.mri.act_mask, "active_ip", hole=self.mri.nec_mask, 
                                       min_value=self.active_min_value, max_value=self.active_max_value,
                                       method=self.interpolation_method)

        ede_ip = self.interpolate_segm(self.mri.ede_mask, "edema_ip", plateau=self.mri.act_mask + self.mri.nec_mask,
                                      min_value=self.edema_min_value, max_value=self.edema_max_value,
                                      method=self.interpolation_method)
        self.mapped_act_file = self.map_field(act_ip, self.fmap_dir + "solid_tumor")
        self.mapped_nec_file = self.map_field(nec_ip, self.fmap_dir + "necrotic")
        self.mapped_ede_file = self.map_field(ede_ip, self.fmap_dir + "edema")

    def generate_wms_map(self):
        work_dir = gen.mkdir_if_not_exist(self.out_dir + "wms_maps" + os.sep)
        # constant white matter at tumour area
        if self.wms_mapping_handler == 0:
            command = [self.wms_dir + "wms_Brain_pve_0.nii.gz"]
            command.append("-add")
            command.append(self.wms_dir + "tmask.nii.gz")
            command.append(work_dir + "wm.nii.gz")
            self.fsl.run_maths(command)
            self.mapped_wm_file = self.map_field(work_dir + "wm.nii.gz", work_dir + "white_matter")
            self.mapped_gm_file = self.map_field(self.wms_dir + "wms_Brain_pve_2.nii.gz", work_dir + "gray_matter")
            self.mapped_csf_file = self.map_field(self.wms_dir + "wms_Brain_pve_1.nii.gz", work_dir + "csf")

    def set_av_params(self, params, distributions, weights):
        return MapAverageMaterialProperty(params, distributions, weights)

