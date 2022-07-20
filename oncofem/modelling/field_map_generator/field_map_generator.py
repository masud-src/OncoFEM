
import nibabel.loadsave
import dolfin
import numpy
import meshio

class FieldMapGenerator:
    def __init__(self):
        pass

    def write_geometry(self, mesh_file, xdmf_dir):
        """
        t.b.d.
        """
        mesh = meshio.read(mesh_file)

        points = mesh.points
        tetra = {"tetra": mesh.cells_dict["tetra"]}

        xdmf = meshio.Mesh(points, tetra)
        meshio.write("%s/geometry.xdmf" % xdmf_dir, xdmf)

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
