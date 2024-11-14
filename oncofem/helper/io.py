"""
Definition of input and output interfaces and post-processing elements

Classes:
    Graph:                      Graph class for outputs of graph plots. Holds neccessary inputs for a single graph and 
                                can be given to time plot for simple output.
    TimePlot:                   Class for creating time plots.

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

from oncofem.helper.general import (add_file_appendix, mkdir_if_not_exist, file_collector,
                                    split_path, get_path_file_extension, ungzip)
import meshio
import dolfin as df
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
import nibabel as nib
import nibabel.loadsave
from skimage import measure
from scipy.ndimage import gaussian_filter
from stl import mesh

class Graph:
    """
    Graph class for outputs of graph plots. Holds neccessary inputs for a single graph
    and can be given to time plot for simple output.

    *Attributes:*
        field: String, name of field
        direction: String, spatial direction of vectors
        dim: Int, dimension of quantity
        label: String, label of graph
        point: int, mesh id of evaluation point
        x_value_list: list of x-values
        y_value_list: list of y-values
        line_color: matplotlib color of graph
        line_style: matplotlib style of graph
        line_width: matplotlib width of graph
        line_marker: matplotlib marker of graph
    """

    def __init__(self):
        self.field = None
        self.direction = None
        self.dim = None
        self.label = None
        self.point = None
        self.x_value_list = None
        self.y_value_list = None
        self.line_color = None
        self.line_style = None
        self.line_width = None
        self.line_marker = None

class TimePlot:
    """
    Class for creating time plots.  

    *Attributes:*
        title: String, name of field
        path: String, hold path for saving
        plot_title: bool if title should be plotted 
        data: list, holds data, that should be plotted
        subtitle: String for subtitle when saving 
        x_label: label of x-dimension
        y_label: label of y-dimension
        plot_legend: bool if legend is plotted (default: True)
        font_size: int, sets font size

    *Methods:*
        plot_data: plots data in set path
        export_legend: exports legend
    """

    def __init__(self, title: str, path: str, plot_title: bool):
        self.title = title
        self.path = path
        self.plot_title = plot_title
        self.data = []
        self.subtitle = None
        self.y_label = None
        self.x_label = None
        self.plot_legend = True
        self.font_size = 10

    def plot_data(self) -> None:
        """
            Plots data into set path
        """
        for dat in self.data:
            plt.plot(dat.x_value_list, dat.y_value_list, c=dat.line_color, 
                     ls=dat.line_style, lw=dat.line_width, marker=dat.line_marker, label=dat.label)
        plt.rcParams['text.usetex'] = True
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.rcParams.update({'font.size': self.font_size})
        plt.rcParams.update({'figure.autolayout': True})
        plt.ticklabel_format(axis="y", style="sci")
        if self.plot_title: 
            plt.title(r"" + self.title)
        if self.plot_legend:
            plt.legend() 
        else:
            self.export_legend(plt.legend(), filename=self.path + os.sep + self.title + "legend.png")
        if self.subtitle is not None:
            plt.savefig(self.path + os.sep + self.title + "-" + self.subtitle)
        else:
            plt.savefig(self.path + os.sep + self.title)
        plt.close()

    def export_legend(self, legend, filename:str="legend.png", expand:list[int]=[-5, -5, 5, 5]) -> None:
        """
            Exports the legend of a time plot entity.

            *Arguments:*
                legend: respective legend
                file_name: String, output file name (default: "legend.png")
                expand: [-x,-y,x,y] List of coordinates for size of image (optional, default = [-5, -5, 5, 5]

            *Example:*
                export_legend(legend, "legend_1.png")
        """
        fig = legend.figure
        fig.canvas.draw()
        bbox = legend.get_window_extent()
        bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
        bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(filename, dpi="figure", bbox_inches=bbox)

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

def nii2stl(filename_nii:str, filename_stl:str, work_dir:str, smoothing_sigma:float=-1.0,
            marching_cubes_levels:float=0.5) -> None:
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

    stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
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

def map_field(field_file: str, mesh: Union[df.Mesh, str], outfile: None=str) -> str:
    """
    Maps field onto mesh file.

    *Arguments*:
        field_file: Nifti file of field
        outfile:    String of output file
        mesh:       Dolfin mesh or path to mesh file

    *Example*:
        xdmf_file = map_field("edema.nii.gz", "edema", mesh)
    """
    image = nibabel.load(field_file)
    data = image.get_fdata()

    if type(mesh) is str:
        mesh = load_mesh(mesh)

    n = mesh.topology().dim()
    regions = df.MeshFunction("double", mesh, n, 0)

    for cell in df.cells(mesh):
        c = cell.index()
        # Convert to voxel space
        ijk = cell.midpoint()[:]
        # Round off to nearest integers to find voxel indices
        i, j, k = np.rint(ijk).astype("int")
        # Insert image data into the mesh function:
        regions.array()[c] = float(data[i, j, k])

    # Store regions in XDMF
    xdmf = df.XDMFFile(mesh.mpi_comm(), outfile + ".xdmf")
    xdmf.parameters["flush_output"] = True
    xdmf.parameters["functions_share_mesh"] = True
    xdmf.write(mesh)
    xdmf.write(regions)
    xdmf.close()
    return outfile + ".xdmf"

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

def load_mesh(file:str) -> df.Mesh:
    """
    Loads an XDMF file from file directory

    *Arguments*:
        file: String of XDMF file directory

    *Example*:
        xdmf_file = load_mesh("studies/test_study/der/geometry/geometry.xdmf")
    """
    mesh = df.Mesh()
    with df.XDMFFile(file) as infile:
        infile.read(mesh)
    return mesh

def read_mapped_xdmf(file:str, field:str="f", value_type:str="double") -> df.MeshFunction:
    """
    Reads a meshfunction from a mapped field in a xdmf file.

    *Arguments*:
        file: String of input file
        field: String, identifier in xdmf file, default: "f"
        value_type: String of type of mapped field, default is double 

    *Example*:
        mesh_function = read_mapped_xdmf("geometry.xdmf")
    """
    mesh = df.Mesh()
    file = df.XDMFFile(file)
    file.read(mesh)
    file.close()
    mvc = df.MeshValueCollection(value_type, mesh, mesh.topology().dim())
    with file as infile:
        infile.read(mvc, field)
    return df.MeshFunction(value_type, mesh, mvc)

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

def read_field_data(path:str) -> list[Graph]:
    """
    Reads field data from a tabbed spaced csv file.

    *Arguments:*
        path: String, path to csv file

    *Example:*
        read_field_data(path)
    """
    dataframe = pd.read_csv(path, sep='\t', lineterminator='\n')
    points = dataframe.iloc[:, 1:-1].columns.values.tolist()
    field_rank = int(dataframe.columns.values[0])
    times = dataframe.iloc[:, 0].values.tolist()
    name = os.path.splitext(os.path.basename(path))[0].split("-")
    if field_rank != 0:
        dim = len(ast.literal_eval(dataframe.iloc[1, 1]))
    else:
        dim = 0

    graphs = []
    for point in points:
        if field_rank == 0:
            graph = Graph()
            graph.dim = dim
            graph.label = ",".join(name[:-1])
            graph.point = int(float(point))
            graph.x_value_list = times
            graph.field = name[-1]
            graph.y_value_list = dataframe.loc[:, str(point)]
            graph.direction = 0
            graphs.append(graph)
        else:
            for i in range(int(dim)):
                graph = Graph()
                graph.dim = dim
                graph.label = str(i) + " " + ",".join(name[:-1])
                graph.point = int(float(point))
                graph.x_value_list = times
                graph.field = name[-1]
                graph.y_value_list = [float(str(
                    str(dataframe.loc[:, str(point)][j]).replace("[", "").replace("]", "").split(" ")[
                        i]).replace(",", "")) for j in range(len(dataframe.loc[:, str(point)]))]
                graph.direction = i
                graphs.append(graph)

    return graphs

def get_data_from_txt_files(file_dir:str) -> list[Graph]:
    """
    Reads out given directory for tab spaced data files
    """
    fileExt = r".txt"
    files = [os.path.join(file_dir, _) for _ in os.listdir(file_dir) if _.endswith(fileExt)]
    return [item for sublist in [read_field_data(file) for file in files] for item in sublist]
