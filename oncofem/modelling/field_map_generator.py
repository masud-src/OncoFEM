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
from oncofem.struc.study import Study
import oncofem.helper.general as gen
import oncofem.interfaces.fsl as fsl
import nibabel.loadsave
import dolfin
import numpy as np
import meshio
import vtk
import SVMTK as svmtk
import nibabel as nib
from skimage.measure import regionprops

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

class TumorMapGenerator:
    def __init__(self, study: Study, working_dir):
        self.study = study
        self.maps_dir = gen.mkdir_if_not_exist(working_dir + "tumor_maps/")
        self.labeled_image = None
        self.orig_field = None
        self.orig_affine = None
        self.out_zero_field = None
        self.labels = None
        self.props = None
        self.solid_tumor_nii = None
        self.necrotic_nii = None
        self.edema_nii = None
        self.max_edema_value = None
        self.max_solid_tumor_value = None
        self.max_necrotic_value = None

#    def write_to_file(self, out, name):
#        gen.mkdir_if_not_exist(self.maps_dir)
#        file = nib.Nifti1Image(out.astype(np.float64), self.orig_affine)
#        nib.save(file, self.maps_dir + name)
#        if os.path.exists(self.maps_dir + name + ".gz"):
#            os.remove(self.maps_dir + name + ".gz")
#        gen.run_shell_command("gzip " + self.maps_dir + name)
#        return self.maps_dir + name + ".gz"

    def read_labelprop_from_image(self, path_to_labeled_image: str):
        self.labeled_image = nib.load(path_to_labeled_image)
        self.orig_affine = self.labeled_image.affine
        # get python array with labels
        self.orig_field = self.labeled_image.get_fdata()
        self.out_zero_field = np.zeros(np.shape(self.orig_field))
        # get bitmask for desired labels
        # must be int type
        self.labels = self.orig_field.astype(int)
        # compute geometrical properties of bitmask
        self.props = regionprops(self.labels, self.orig_field)

    def generate_solid_tumor_map(self):
        out = self.out_zero_field
        voxels_tumor = self.props[2].coords
        for i, voxel in enumerate(voxels_tumor):
            out[voxel[0], voxel[1], voxel[2]] = self.max_solid_tumor_value #TODO: Fix this value

        self.solid_tumor_nii = self.write_to_file(out, "solid_tumor_map.nii")

    def generate_necrotic_tumor_map(self):
        out = self.out_zero_field
        voxels_necrotic = self.props[0].coords
        for i, voxel in enumerate(voxels_necrotic):
            out[voxel[0], voxel[1], voxel[2]] = self.max_necrotic_value #TODO: Fix this value

        self.necrotic_nii = self.write_to_file(out, "necrotic_core_map.nii")

    def generate_edema_map(self):
        necrotic_voxels = self.props[0].coords
        necrotic_centroid = self.props[0].centroid
        necrotic_area = self.props[0].area
        necrotic_weight = 1
        solid_tumor_voxels = self.props[2].coords
        solid_tumor_centroid = self.props[2].centroid
        solid_tumor_area = self.props[2].area
        solid_tumor_weight = 1
        edema_voxels = self.props[1].coords
        edema_centroid = self.props[1].centroid
        edema_area = self.props[1].area
        edema_weight = 1

        momentum_x = necrotic_centroid[0] * necrotic_weight * necrotic_area + solid_tumor_centroid[0] * solid_tumor_weight * solid_tumor_area + edema_centroid[0] * edema_weight * edema_area 
        momentum_y = necrotic_centroid[1] * necrotic_weight * necrotic_area + solid_tumor_centroid[1] * solid_tumor_weight * solid_tumor_area + edema_centroid[1] * edema_weight * edema_area 
        momentum_z = necrotic_centroid[2] * necrotic_weight * necrotic_area + solid_tumor_centroid[2] * solid_tumor_weight * solid_tumor_area + edema_centroid[2] * edema_weight * edema_area 
        weights = necrotic_weight * necrotic_area + solid_tumor_weight * solid_tumor_area + edema_weight * edema_area 

        overall_centroid = np.asarray([momentum_x / weights, momentum_y / weights, momentum_z / weights])
        dist_voxel_center_edema = [np.linalg.norm(voxel - overall_centroid) for voxel in edema_voxels]

        max_dist_voxel_edema = max(dist_voxel_center_edema)
        normed_dist = np.zeros((len(dist_voxel_center_edema), 1))
        for i, dist_voxel in enumerate(dist_voxel_center_edema):
            x = dist_voxel / max_dist_voxel_edema * np.pi
            normed_dist[i] = np.sqrt(2) / (0.798 * np.sqrt(np.pi)) * np.exp(-x * x / (0.798 * 0.798 * 2.0))

        out = self.out_zero_field
        for i, voxel in enumerate(edema_voxels):
            out[voxel[0], voxel[1], voxel[2]] = normed_dist[i] * self.max_edema_value #TODO: Fix this value

        #for i, voxel in enumerate(necrotic_voxels):
        #    out[voxel[0], voxel[1], voxel[2]] = 0

        #for i, voxel in enumerate(solid_tumor_voxels):
        #    out[voxel[0], voxel[1], voxel[2]] = 0

        self.edema_nii = self.write_to_file(out, "edema_map.nii")

class FieldMapGenerator:
    def __init__(self, study: Study):
        self.study = study
        self.t1_dir = None
        self.work_dir = None
        self.out_dir = None
        self.wms_dir = None
        self.geom_stl_file = None
        self.geom_mesh_file = None
        self.geom_xdmf_file = None
        self.surf_xdmf_file = None
        self.tumor_seg_file = None
        self.mesh = None
        self.tumor_mapping_handler = 0
        self.wms_mapping_handler = 0
        self.volume_resolution = 16
        self.tmg = None
        self.mapped_edema_file = None
        self.mapped_solid_tumor_file = None
        self.mapped_necrotic_file = None
        self.mapped_wm_file = None
        self.mapped_gm_file = None
        self.mapped_csf_file = None
        self.fsl = fsl.FSL()

    def set_general(self, t1_dir, work_dir):
        self.t1_dir = t1_dir
        self.work_dir = work_dir
        self.out_dir = gen.mkdir_if_not_exist(self.work_dir + "fmap" + os.sep)
        self.geom_stl_file = self.out_dir + "geometry.stl"
        self.geom_mesh_file = self.out_dir + "geometry.mesh"

    def nii2stl(self, filename_nii, filename_stl, label):
        """
        https://github.com/MahsaShk/MeshProcessing
        Read a nifti file including a binary map of a segmented organ with label id = label. 
        Convert it to a smoothed mesh of type stl.
        filename_nii     : Input nifti binary map 
        filename_stl     : Output mesh name in stl format
        label            : segmented label id 
        """
        if filename_nii.endswith(".gz"):
            path, file, file_wo = gen.get_path_file_extension(filename_nii)
            t1_ungzip = self.work_dir + file_wo + ".nii"
            t1_dir = filename_nii
            gen.ungzip(t1_dir, t1_ungzip)
            filename_nii = t1_ungzip

        # read the file
        reader = vtk.vtkNIFTIImageReader()
        reader.SetFileName(filename_nii)
        reader.Update()

        # apply marching cube surface generation
        surf = vtk.vtkDiscreteMarchingCubes()
        surf.SetInputConnection(reader.GetOutputPort())
        surf.SetValue(0, label)  # use surf.GenerateValues function if more than one contour is available in the file
        surf.Update()

        # smoothing the mesh
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        if vtk.VTK_MAJOR_VERSION <= 5:
            smoother.SetInput(surf.GetOutput())
        else:
            smoother.SetInputConnection(surf.GetOutputPort())
        smoother.SetNumberOfIterations(30)
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()  # The positions can be translated and scaled such that they fit within a range of [-1, 1] prior to the smoothing computation
        smoother.GenerateErrorScalarsOn()
        smoother.Update()

        # save the output
        writer = vtk.vtkSTLWriter()
        writer.SetInputConnection(smoother.GetOutputPort())
        writer.SetFileTypeToASCII()
        writer.SetFileName(filename_stl)
        writer.Write()

    def stl2mesh(self, stl_file, mesh_file, resolution=16):
        # Load input file
        surface = svmtk.Surface(stl_file)
        # Generate the volume mesh
        domain = svmtk.Domain(surface)
        domain.create_mesh(resolution)
        # Write the mesh to the output file
        domain.save(mesh_file)

    def mesh2xdmf(self, mesh_file, xdmf_dir):
        """
        t.b.d.
        """
        mesh = meshio.read(mesh_file)
        points = mesh.points
        tetra = {"tetra": mesh.cells_dict["tetra"]}
        xdmf_geom = meshio.Mesh(points, tetra)
        meshio.write("%s/geometry.xdmf" % xdmf_dir, xdmf_geom)
        self.geom_xdmf_file = xdmf_dir + "geometry.xdmf"

    def generate_geometry_file(self):
        # first nii2stl
        self.nii2stl(self.t1_dir, self.geom_stl_file, 0)
        # second stl2mesh
        self.stl2mesh(self.geom_stl_file, self.geom_mesh_file, self.volume_resolution)
        # third msh2xmdf
        self.mesh2xdmf(self.geom_mesh_file, self.out_dir)

    def set_fixed_boundary(self, x_bounds=None, y_bounds=None, z_bounds=None):
        mesh = dolfin.Mesh()
        with dolfin.XDMFFile(self.geom_xdmf_file) as infile:
            infile.read(mesh)
        mf_domain = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim(),0)
        mf_facet = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim()-1)
        BoundingBox(mesh, x_bounds, y_bounds, z_bounds).mark(mf_facet, 1)
        self.surf_xdmf_file = self.out_dir + "surface.xdmf"
        dolfin.XDMFFile.write(dolfin.XDMFFile(self.surf_xdmf_file), mf_facet)
        return mf_domain, mf_facet

    def read_mapped_xdmf(self, ip, value_type: str = "double"):
        mesh = dolfin.Mesh()
        file = dolfin.XDMFFile(ip)
        file.read(mesh)
        file.close()
        mvc = dolfin.MeshValueCollection(value_type, mesh, mesh.topology().dim())
        with dolfin.XDMFFile(ip) as infile:
            infile.read(mvc, "f")
        return dolfin.MeshFunction(value_type, mesh, mvc)

    def meshfunction_2_function(self, mf: dolfin.MeshFunction, fs: dolfin.FunctionSpace):
        """maps meshfunction to functionspace. Only works with constant meshfunction space and linear functionspace"""
        v2d = dolfin.vertex_to_dof_map(fs)
        u = dolfin.Function(fs)
        u.vector()[v2d] = mf.array()
        return u

    def set_up_tumor_map_generator(self):
        self.tmg = TumorMapGenerator(self.study, self.out_dir)
        self.tmg.read_labelprop_from_image(self.tumor_seg_file)
        return self.tmg

    def generate_tumor_map(self):
        # Needed to change edema with necrotic...somehow lead to overwriting of edema
        # generate separated nii maps
        self.tmg.generate_solid_tumor_map()
        self.tmg.generate_edema_map()
        self.tmg.generate_necrotic_tumor_map()
        # generate xdmf files
        self.mapped_solid_tumor_file = self.map_field(self.tmg.solid_tumor_nii, self.tmg.maps_dir + "solid_tumor")
        self.mapped_edema_file = self.map_field(self.tmg.edema_nii, self.tmg.maps_dir + "edema")
        self.mapped_necrotic_file = self.map_field(self.tmg.necrotic_nii, self.tmg.maps_dir + "necrotic")

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

    def remesh_surface(self, stl_input, output, max_edge_length, n, do_not_move_boundary_edges=False):
        surface = svmtk.Surface(stl_input)
        surface.isotropic_remeshing(max_edge_length, n, do_not_move_boundary_edges)
        surface.save(output)

    def smoothen_surface(self, stl_input, output, n=1, eps=1.0, preserve_volume=True):
        surface = svmtk.Surface(stl_input)
        if preserve_volume:
            surface.smooth_taubin(n)
        else:
            surface.smooth_laplacian(eps, n)
        surface.save(output)

    def map_field(self, field_file, outfile, mesh_file=None):
        """
        t.b.d.
        """
        if mesh_file is None:
            mesh_file = self.geom_xdmf_file
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
