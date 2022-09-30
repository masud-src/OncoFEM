"""
field map generator
"""
import os

from oncofem.struct.study import Study
from oncofem.modelling.field_map_generator.tumor_map_generator import TumorMapGenerator
from oncofem.helper.general import ungzip, get_path_file_extension, set_working_folder
import nibabel.loadsave
import dolfin
import numpy as np
import meshio
import vtk
import SVMTK as svmtk


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
        self.volume_resolution = 16

    def set_general(self, t1_dir, work_dir):
        self.t1_dir = t1_dir
        self.work_dir = work_dir
        self.out_dir = set_working_folder(self.work_dir + "fmap" + os.sep)
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
            path, file, file_wo = get_path_file_extension(filename_nii)
            t1_ungzip = self.work_dir + file_wo + ".nii"
            t1_dir = filename_nii
            ungzip(t1_dir, t1_ungzip)
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


    def generate_tumor_map(self):
        tmg = TumorMapGenerator(self.study, self.out_dir)
        tmg.read_labelprop_from_image(self.tumor_seg_file)
        # generate separated nii maps
        tmg.generate_solid_tumor_map()
        tmg.generate_necrotic_tumor_map()
        tmg.generate_edema_map()
        # generate xdmf files
        self.map_field(tmg.solid_tumor_nii, tmg.maps_dir + "solid_tumor.xdmf")
        self.map_field(tmg.necrotic_nii, tmg.maps_dir + "necrotic.xdmf")
        self.map_field(tmg.edema_nii, tmg.maps_dir + "edema.xdmf")

    def generate_wms_map(self):
        work_dir = set_working_folder(self.out_dir + "wms_maps" + os.sep)
        self.map_field(self.wms_dir + "wms_Brain_pve_0.nii.gz", work_dir + "wm_Brain.xdmf")
        self.map_field(self.wms_dir + "wms_Brain_pve_1.nii.gz", work_dir + "csf_Brain.xdmf")
        self.map_field(self.wms_dir + "wms_Brain_pve_2.nii.gz", work_dir + "gm_Brain.xdmf")
        self.map_field(self.wms_dir + "wms_Tumor_pve_0.nii.gz", work_dir + "nec_Tumor.xdmf")
        self.map_field(self.wms_dir + "wms_Tumor_pve_1.nii.gz", work_dir + "act_Tumor.xdmf")
        self.map_field(self.wms_dir + "wms_Tumor_pve_2.nii.gz", work_dir + "ede_Tumor.xdmf")

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
            mesh_file=self.geom_xdmf_file
        image = nibabel.load(field_file)
        data = image.get_fdata()

        mesh = dolfin.Mesh()
        with dolfin.XDMFFile(mesh_file) as file:
            file.read(mesh)

        n = mesh.topology().dim()
        regions = dolfin.MeshFunction("size_t", mesh, n, 0)

        for cell in dolfin.cells(mesh):
            c = cell.index()

            # Convert to voxel space
            ijk = cell.midpoint()[:]

            # Round off to nearest integers to find voxel indices
            i, j, k = np.rint(ijk).astype("int")

            # Insert image data into the mesh function:
            regions.array()[c] = int(data[i, j, k])

        # Store regions in XDMF
        xdmf = dolfin.XDMFFile(mesh.mpi_comm(), outfile + ".xdmf")
        xdmf.parameters["flush_output"] = True
        xdmf.parameters["functions_share_mesh"] = True
        xdmf.write(mesh)
        xdmf.write(regions)
        xdmf.close()
