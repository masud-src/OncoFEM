"""
Definition of input and output interfaces and post-processing elements

Functions:
    msh2xdmf:                   Generates from input msh file two or three output files in xdmf format for input into 
                                FEniCS. Output files are: (tetra.xdmf), triangle.xdmf, lines.xmdf
    getXDMF:                    Gathers all needed input files from a respective folder in workingdata environment and 
                                returns the files in the following order: (tetra.xdmf), triangle.xdmf, lines.xmdf
    set_output_file:            Initializes xdmf file of given name. That file can be filled with multiple fields using 
                                the same mesh
    nii2stl:                    https://github.com/MahsaShk/MeshProcessing, Read a nifti file including a binary map of 
                                a segmented organ with label id = label. Convert it to a smoothed mesh of type stl.
    stl2mesh:                   https://github.com/SVMTK/SVMTK, Converts a stl surface file into a mesh volume file. 
    map_field:                  Maps field onto mesh file. Optionally a different mesh_file can be chosen 
    mesh2xdmf:                  Converts a mesh file into a xdmf file.
    load_mesh:                  Loads an XDMF file from file directory
    read_mapped_xdmf:           Reads a meshfunction from a mapped field in a xdmf file.
    remesh_surface:             https://github.com/kent-and/mri2fem, Remeshes the surface of a stl surface mesh.
    smoothen_surface:           https://github.com/kent-and/mri2fem, Smoothes the surface of a stl surface mesh.
    write_field2xdmf:           writes field to outputfile, also can write nodal values into separated txt-files. 
                                Therefore, list of nodal id's and mesh should be given. In case of non-scalar fields, 
                                field_dim should be given.
    write_field2nii:            writes field to outputfile, also can write nodal values into separated txt-files. 
                                Therefore, list of nodal id's and mesh should be given. In case of non-scalar fields, 
                                field_dim should be given.
    read_field_data:            Reads field data from a tabbed spaced csv file.
    get_data_from_txt_files:    Reads out given directory for tab spaced data files
"""
from typing import Union

from oncofem.utils.general import (add_file_appendix, mkdir_if_not_exist, file_collector,
                                   split_path, get_path_file_extension, ungzip)
import meshio
import dolfin as df
import os
import numpy as np
import nibabel as nib
import nibabel.loadsave
from skimage import measure
from scipy.ndimage import gaussian_filter
from stl import mesh as npmesh

def msh2xdmf(inputfile: Union[str, meshio.Mesh], outputfolder: str, correct_gmsh:bool=False) -> bool:
    """
    Generates from input msh file two or three output files in xdmf format for input into FEniCS. Output files are:
    (tetra.xdmf), triangle.xdmf, lines.xmdf

    *Arguments:*
        inputfile: input_msh_file
        outputfolder: output_folder

    *Example:*
        msh2xdmf("inputdata/Terzaghi.msh", "Terzaghi_2d")
    """
    if type(inputfile) is str:
        inputfile = add_file_appendix(inputfile, "msh")
        mkdir_if_not_exist(outputfolder)
        msh = meshio.read(inputfile)
    else:
        msh = inputfile

    cells = {"tetra": None, "triangle": None, "line": None, "vertex": None}
    data = {"tetra": None, "triangle": None, "line": None, "vertex": None}

    for cell in msh.cells:
        if cells[cell.type] is None:
            cells[cell.type] = cell.data
        else:
            cells[cell.type] = np.vstack([cells[cell.type], cell.data])

    for key in msh.cell_data_dict["gmsh:physical"].keys():
        if data[key] is None:
            data[key] = msh.cell_data_dict["gmsh:physical"][key]
        else:
            data[key] = np.vstack([data[key], msh.cell_data_dict["gmsh:physical"][key]])

    if correct_gmsh:
        points = np.zeros((len(msh.points), 2))
        for i, point in enumerate(msh.points):
            points[i] = [point[0], point[1]]
    else:
        points = msh.points

    for key in cells:
        if cells[key] is not None:
            print("write ", key, "_mesh")
            mesh = meshio.Mesh(points=points, cells={key: cells[key]}, cell_data={"name_to_read": [data[key]]})
            meshio.write(outputfolder + os.sep + str(key) + ".xdmf", mesh)
    return True


# noinspection PyBroadException
def getXDMF(inputdirectory: str) -> list[df.XDMFFile]:
    """
    Gathers all needed input files from a respective folder in workingdata environment and returns 
    the files in the following order: (tetra.xdmf), triangle.xdmf, lines.xmdf

    *Arguments:*
        inputdirectory: input_folder

    *Example:*
        getXDMF("Terzaghi_2d")
    """
    input_files = [split_path(input_file)[0] for input_file in list(file_collector(inputdirectory, "xdmf"))]
    keys = {"tetra.xdmf": 0, "triangle.xdmf": 1, "line.xdmf": 2, "point.xdmf": 3}
    xdmf_files = [None] * 4
    mesh = df.Mesh()

    if "tetra.xdmf" in input_files:
        with df.XDMFFile(inputdirectory + "/tetra.xdmf") as infile:
            infile.read(mesh)
    elif "triangle.xdmf" in input_files:
        with df.XDMFFile(inputdirectory + "/triangle.xdmf") as infile:
            infile.read(mesh)

    for file in input_files:
        mvc = df.MeshValueCollection("size_t", mesh, mesh.topology().dim())
        with df.XDMFFile(inputdirectory + os.sep + file) as infile:
            infile.read(mvc, "name_to_read")
        mf = df.MeshFunction("size_t", mesh, mvc)
        xdmf_files[keys[file]] = mf

    return filter(None, xdmf_files)


def set_output_file(name: str) -> df.XDMFFile:
    """
    Initializes xdmf file of given name. That file can be filled with multiple fields using the same mesh

    *Arguments*
        name: File path    

    *Example*
        output_file = set_output_file("solution/3d_crashtest.xdmf")
    """
    xdmf_file = df.XDMFFile(name + ".xdmf")
    xdmf_file.rename(name, "x")
    xdmf_file.parameters["flush_output"] = True
    xdmf_file.parameters["functions_share_mesh"] = True
    return xdmf_file


def nii2stl(filename_nii: str, filename_stl: str, work_dir: str, smoothing_sigma: float = -1.0,
            marching_cubes_levels: float = 0.5) -> None:
    """
    Read a nifti file including a binary map of a segmented organ with label id = label.
    Convert it to a smoothed mesh of type stl.
    :param filename_nii: Input nifti file
    :param filename_stl: Output stl file
    :param work_dir: Working direction
    :param smoothing_sigma: smoothing parameter for gaussian filter (-1.0 for off)
    :param marching_cubes_levels: Value that should be extracted. If any value between 1
           and 0 it should be set to 0.5. For segmented value it should be set to the
           particular value
    :return: None
    """
    if filename_nii.endswith(".gz"):
        path, file, file_wo = get_path_file_extension(filename_nii)
        t1_ungzip = work_dir + file_wo + ".nii"
        t1_dir = filename_nii
        ungzip(t1_dir, t1_ungzip)
        filename_nii = t1_ungzip

    nifti_img = nib.load(filename_nii)
    data = nifti_img.get_fdata()
    if smoothing_sigma != -1.0:
        data = gaussian_filter(data.astype(np.float32), sigma=smoothing_sigma)
    verts, faces, _, _ = measure.marching_cubes(data, level=marching_cubes_levels)

    stl_mesh = npmesh.Mesh(np.zeros(faces.shape[0], dtype=npmesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            stl_mesh.vectors[i][j] = verts[face[j], :]

    stl_mesh.save(filename_stl)

def stl2mesh(stl_file: str, mesh_file: str, resolution: int = 16) -> None:
    """
    https://github.com/SVMTK/SVMTK
    Converts a stl surface file into a mesh volume file. 

    *Arguments*:
        stl_file: String of input file in stl format
        mesh_file: String of output file in mesh format
        resolution: int of mesh resolution

    *Example*:
        stl2mesh("input.stl", "output.mesh", 16)
    """
    try:
        import SVMTK as svmtk
    except ImportError:
        print("In order to handle stl files, SVMTK need to be installed. Please install it to proceed.")
    surface = svmtk.Surface(stl_file)
    domain = svmtk.Domain(surface)
    domain.create_mesh(resolution)
    domain.save(mesh_file)

def remesh_surface(stl_input: str, output: str, max_edge_length: float, 
                   n: int, do_not_move_boundary_edges: bool = False) -> None:
    """
    https://github.com/kent-and/mri2fem
    Remeshes the surface of a stl surface mesh. Taken from mri2fem.

    *Arguments*:
        stl_input: String of stl input file
        output: String of output file
        max_edge_length: Float, maximum length of element edge
        n: int, remesh iterations
        do_not_move_boundary_edges: fixes boundary edges

    *Example*:
        remesh_surface("geometry.stl", "geometry_remesh.stl", 1.0, 3)
    """
    try:
        import SVMTK as svmtk
    except ImportError:
        print("In order to handle stl files, SVMTK need to be installed. Please install it to proceed.")
    surface = svmtk.Surface(stl_input)
    surface.isotropic_remeshing(max_edge_length, n, do_not_move_boundary_edges)
    surface.save(output)

def smoothen_surface(stl_input:str, output:str, n:int=1, eps:float=1.0, preserve_volume:bool=True) -> None:
    """"
    https://github.com/kent-and/mri2fem
    Smoothes the surface of a stl surface mesh. Taken from mri2fem.

    *Arguments*:
        stl_input: String of stl input file
        output: String of output file
        n: int, smoothing iterations
        eps: float, smoothing factor
        preserve_volume: fixes volume

    *Example*:
        smoothen_surface("geometry.stl", "geometry_smooth.stl", n=10, eps=1.0, preserve_volume=False)
    """
    try:
        import SVMTK as svmtk
    except ImportError:
        print("In order to handle stl files, SVMTK need to be installed. Please install it to proceed.")
    surface = svmtk.Surface(stl_input)
    if preserve_volume:
        surface.smooth_taubin(n)
    else:
        surface.smooth_laplacian(eps, n)
    surface.save(output)


def mesh2xdmf(mesh_file:str, xdmf_dir:str) -> str:
    """
    converts a mesh file into a xdmf file.

    *Arguments*:
        mesh_file: String of input mesh file
        xdmf_dir: String of output directory for "geometry.xdmf" file

    *Example*:
        xdmf_dir = mesh2xdmf("geometry.mesh", "studies/test_study/der/geometry/")
    """
    mesh = meshio.read(mesh_file)
    points = mesh.points
    tetra = {"tetra": mesh.cells_dict["tetra"]}
    xdmf_geom = meshio.Mesh(points, tetra)
    meshio.write("%s/geometry.xdmf" % xdmf_dir, xdmf_geom)
    return xdmf_dir + "geometry.xdmf"


def write_field2xdmf(outputfile:df.XDMFFile, field:df.Function, fieldname:str, timestep:float, 
                     function_space:df.FunctionSpace=None, id_nodes:list[int]=None, mesh:df.Mesh=None) -> list[list]:
    """
    writes field to outputfile, also can write nodal values into separated txt-files. 
    Therefore, list of nodal id's and mesh should be given.
    In case of non-scalar fields, field_dim should be given.

    *Arguments:*
        outputfile: xdmf_file
        field: scalar, vector or tensor-valued field
        fieldname: String
        timestep: respective timestep
        id_nodes: list of node identifiers 
        mesh: respective mesh

    *Example:*
        xdmf_file = write_field2output(xdmf_file, u, "displacement", t)
    """
    if type(field) is not df.Function:
        field = df.project(field, function_space, solver_type="cg")
    field.rename(fieldname, fieldname)
    outputfile.write(field, timestep)
    if id_nodes is not None:
        if timestep == 0:
            with open(outputfile.name() + "-" + fieldname + ".txt", "w") as myfile:
                myfile.write(str(field.value_rank()) + "\t")
                for node in id_nodes:
                    myfile.write(str(node) + "\t")
                myfile.write("\n")
                myfile.write(str(timestep) + "\t")
                for node in id_nodes:
                    myfile.write(str(field(mesh.coordinates()[node])) + "\t")
                myfile.write("\n")
        else:
            with open(outputfile.name() + "-" + fieldname + ".txt", "a") as myfile:
                myfile.write(str(timestep) + "\t")
                for node in id_nodes:
                    myfile.write(str(field(mesh.coordinates()[node]).tolist()) + "\t")
                myfile.write("\n")
        return [[field(mesh.coordinates()[node]), node] for node in id_nodes]


def set_out_dir(parent: str, child: str) -> str:
    """
    checks if parent path has separator at end and merges the paths.

    :param parent: String of parent directory
    :param child: String of child directory
    :return: String of combined path
    """
    if not parent.endswith(os.sep):
        parent = parent + os.sep
    return parent + child


def write_field2nii(field:np.ndarray, file_name:str, affine:np.ndarray, t:float=None) -> str:
    """
    writes field to outputfile, also can write nodal values into separated txt-files. 
    Therefore, list of nodal id's and mesh should be given.
    In case of non-scalar fields, field_dim should be given.

    *Arguments:*
        field: scalar, vector or tensor-valued field
        t: Float, time
        field_name: String
        file_name: String
        affine: nib.affine,
        header: nib.header
        type: String (optional, default="nii")

    *Example:*
        write_field2nii(field, t, u, "displacement", field.affine, field.header)
    """
    img = nib.Nifti1Image(field, affine)
    if t is None:
        nib.save(img, file_name + ".nii.gz")
    else:
        nib.save(img, file_name + "_" + str(t) + ".nii.gz")
    return file_name + "_" + str(t) + ".nii.gz"
