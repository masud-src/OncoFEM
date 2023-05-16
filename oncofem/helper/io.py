"""
Definition of input and output interface from gmsh to FEniCS

Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""

from oncofem.helper.general import add_file_appendix, mkdir_if_not_exist, file_collector, splitPath
import meshio
import dolfin as df
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
import nibabel as nib


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

    def plot_data(self):
        """
            Plots data into set path
        """
        for dat in self.data:
            plt.plot(dat.x_value_list, dat.y_value_list, c=dat.line_color, ls=dat.line_style, lw=dat.line_width, marker=dat.line_marker, label=dat.label)
        plt.rcParams['text.usetex'] = True
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.rcParams.update({'font.size': self.font_size})
        plt.rcParams.update({'figure.autolayout': True})
        plt.ticklabel_format(axis="y", style="sci")
        if self.plot_title: plt.title(r"" + self.title)
        # plt.legend() if self.plot_legend else self.export_legend(plt.legend(), filename=self.path + os.sep + self.title + "legend.png")
        if self.subtitle is not None:
            plt.savefig(self.path + os.sep + self.title + "-" + self.subtitle)
        else:
            plt.savefig(self.path + os.sep + self.title)
        plt.close()

    def export_legend(self, legend, filename="legend.png", expand=[-5, -5, 5, 5]):
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


def msh2xdmf(inputfile, outputfolder):
    """
    Generates from input msh file two or three output files in xdmf format for input into FEniCS. Output files are:
    (tetra.xdmf), triangle.xdmf, lines.xmdf

    *Arguments:*
        inputfile: input_msh_file
        outputfolder: output_folder

    *Example:*
        msh2xdmf("inputdata/Terzaghi.msh", "Terzaghi_2d")
    """
    inputfile = add_file_appendix(inputfile, "msh")
    mkdir_if_not_exist(outputfolder, False)
    msh = meshio.read(inputfile)

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

    for key in cells:
        if cells[key] is not None:
            print("write ", key, "_mesh")
            mesh = meshio.Mesh(points=msh.points, cells={key: cells[key]}, cell_data={"name_to_read": [data[key]]})
            meshio.write(outputfolder + os.sep + str(key) + ".xdmf", mesh)
    return True


# noinspection PyBroadException
def getXDMF(inputdirectory):
    """
    Gathers all needed input files from a respective folder in workingdata environment and returns 
    the files in the following order: (tetra.xdmf), triangle.xdmf, lines.xmdf

    *Arguments:*
        inputdirectory: input_folder

    *Example:*
        getXDMF("Terzaghi_2d")
    """
    input_files = [splitPath(input_file)[0] for input_file in list(file_collector(inputdirectory, "xdmf"))]
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

def set_output_file(name: str):
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


# TODO: Check if write to outputfile can be combined! Maybe with nii2mesh!

def write_field2xdmf(outputfile: df.XDMFFile, field: df.Function, fieldname: str, timestep: float, function_space=None, id_nodes=None, mesh=None):
    """
    writes field to outputfile, also can write nodal values into separated txt-files. Therefore, list of nodal id's and mesh should be given.
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


def write_field2nii(field, t, file_name: str, affine, type="nii"):
    """
    writes field to outputfile, also can write nodal values into separated txt-files. Therefore, list of nodal id's and mesh should be given.
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
    if type == "nii":
        img = nib.Nifti1Image(field, affine)
        nib.save(img, file_name + "_" + str(t) + ".nii.gz")
        return file_name + "_" + str(t) + ".nii.gz"
    elif type == "xdmf":
        # check with fieldmapgenerator
        # map_field(self, field_file, outfile, mesh_file=None)
        pass

def read_field_data(path: str):
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

def get_data_from_txt_files(file_dir: str):
    """
    Reads out given directory for tab spaced data files
    """
    fileExt = r".txt"
    files = [os.path.join(file_dir, _) for _ in os.listdir(file_dir) if _.endswith(fileExt)]
    return [item for sublist in [read_field_data(file) for file in files] for item in sublist]
