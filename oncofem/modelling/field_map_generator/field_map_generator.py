"""
field map generator
"""
import os

from oncofem.struct.study import Study
from oncofem.helper.general import ungzip, get_path_file_extension
import nibabel.loadsave
import dolfin
import numpy
import meshio
import vtk
import SVMTK as svmtk

class FieldMapGenerator:
    def __init__(self, study: Study):
        self.study = study
        self.t1_dir = None
        self.work_dir = None
        self.geom_stl = None
        self.geom_mesh = None
        self.geom_xdmf = None
        self.volume_resolution = 16

    def set_general(self, t1_dir, work_dir):
        self.t1_dir = t1_dir
        self.work_dir = work_dir
        self.geom_stl = work_dir + "geometry.stl"
        self.geom_mesh = work_dir + "geometry.mesh"

    def generate_geometry_file(self):
        print("generate_geometry_file")
        #first nii2stl
        self.nii2stl(self.t1_dir, self.geom_stl, 0)
        #second stl2mesh
        self.stl2mesh(self.geom_stl, self.geom_mesh, self.volume_resolution)
        #third msh2xmdf
        self.mesh2xdmf(self.geom_mesh, self.work_dir)

    def stl2mesh(self, stl_file, mesh_file, resolution=16):
        # Load input file
        surface = svmtk.Surface(stl_file)
        # Generate the volume mesh
        domain = svmtk.Domain(surface)
        domain.create_mesh(resolution)
        # Write the mesh to the output file
        domain.save(mesh_file)

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

    def mesh2xdmf(self, mesh_file, xdmf_dir):
        """
        t.b.d.
        """
        mesh = meshio.read(mesh_file)
        points = mesh.points
        tetra = {"tetra": mesh.cells_dict["tetra"]}
        xdmf = meshio.Mesh(points, tetra)
        meshio.write("%s/geometry.xdmf" % xdmf_dir, xdmf)
        self.geom_xdmf = xdmf_dir+os.sep + "geometry.xdmf"

    def map_field(self, field_file, mesh_file, outfile):
        """
        t.b.d.
        """
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
            i, j, k = numpy.rint(ijk).astype("int")

            # Insert image data into the mesh function:
            regions.array()[c] = int(data[i, j, k])

        # Store regions in XDMF
        xdmf = dolfin.XDMFFile(mesh.mpi_comm(), outfile + ".xdmf")
        xdmf.parameters["flush_output"] = True
        xdmf.parameters["functions_share_mesh"] = True
        xdmf.write(mesh)
        xdmf.write(regions)
        xdmf.close()

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
