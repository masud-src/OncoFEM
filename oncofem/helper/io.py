"""
# **************************************************************************#
#                                                                           #
# === IO ===================================================================#
#                                                                           #
# **************************************************************************#
# Definition of input and output interface from gmsh to FEniCS
#
# Co-author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
# Co-author: Maximilian Brodbeck <maximilian.brodbeck@isd.uni-stuttgart.de>
#
# --------------------------------------------------------------------------#
"""

import meshio
from dolfin import Mesh, MeshValueCollection, XDMFFile, MeshFunction, Function
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
import pylab

# **************************************************************************#
#      Classes                                                              #
# **************************************************************************#
# Definition of Classes

# --------------------------------------------------------------------------#

class Graph():
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

class TimePlot():

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
        for dat in self.data:
            plt.plot(dat.x_value_list, dat.y_value_list, c=dat.line_color, ls=dat.line_style, lw=dat.line_width, marker=dat.line_marker, label=dat.label)
        plt.rcParams['text.usetex'] = True
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.rcParams.update({'font.size': self.font_size})
        plt.rcParams.update({'figure.autolayout': True})
        plt.ticklabel_format(axis="y",style="sci")
        if self.plot_title: plt.title(r""+self.title)
        #plt.legend() if self.plot_legend else self.export_legend(plt.legend(), filename=self.path + os.sep + self.title + "legend.png")
        if self.subtitle is not None: plt.savefig(self.path + os.sep + self.title + "-" + self.subtitle)
        else: plt.savefig(self.path + os.sep + self.title)
        plt.close()

    def export_legend(self, legend, filename="legend.png", expand=[-5, -5, 5, 5]):
        fig = legend.figure
        fig.canvas.draw()
        bbox = legend.get_window_extent()
        bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
        bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(filename, dpi="figure", bbox_inches=bbox)


# **************************************************************************#
#      Functions                                                            #
# **************************************************************************#
# Definition of Functions

# --------------------------------------------------------------------------#
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
    if not inputfile.endswith(".msh"): inputfile += ".msh"
    try:
        Path(outputfolder).mkdir(parents=True, exist_ok=False)
    except (FileExistsError):
        print("Folder already exists")

    msh = meshio.read(inputfile)
    point_cells = []
    line_cells = []
    triangle_cells = []
    tetra_cells = []
    for cell in msh.cells:
        if cell.type == "tetra":
            if len(tetra_cells) == 0:
                tetra_cells = cell.data
            else:
                tetra_cells = np.vstack([tetra_cells, cell.data])
        elif cell.type == "triangle":
            if len(triangle_cells) == 0:
                triangle_cells = cell.data
            else:
                triangle_cells = np.vstack([triangle_cells, cell.data])
        elif cell.type == "line":
            if len(line_cells) == 0:
                line_cells = cell.data
            else:
                line_cells = np.vstack([line_cells, cell.data])
        elif cell.type == "vertex":
            if len(point_cells) == 0:
                point_cells = cell.data
            else:
                point_cells = np.vstack([point_cells, cell.data])

    point_data = []
    line_data = []
    triangle_data = []
    tetra_data = []
    for key in msh.cell_data_dict["gmsh:physical"].keys():
        if key == "vertex":
            if len(point_data) == 0:
                point_data = msh.cell_data_dict["gmsh:physical"][key]
            else:
                point_data = np.vstack([point_data, msh.cell_data_dict["gmsh:physical"][key]])
        elif key == "line":
            if len(line_data) == 0:
                line_data = msh.cell_data_dict["gmsh:physical"][key]
            else:
                line_data = np.vstack([line_data, msh.cell_data_dict["gmsh:physical"][key]])
        elif key == "triangle":
            if len(triangle_data) == 0:
                triangle_data = msh.cell_data_dict["gmsh:physical"][key]
            else:
                triangle_data = np.vstack([triangle_data, msh.cell_data_dict["gmsh:physical"][key]])
        elif key == "tetra":
            if len(tetra_data) == 0:
                tetra_data = msh.cell_data_dict["gmsh:physical"][key]
            else:
                tetra_data = np.vstack([tetra_data, msh.cell_data_dict["gmsh:physical"][key]])
    if len(tetra_cells)!=0:
        print("write tetra_mesh")
        tetra_mesh = meshio.Mesh(points=msh.points, cells={"tetra": tetra_cells}, cell_data={"name_to_read": [tetra_data]})
        meshio.write(outputfolder + "/tetra.xdmf", tetra_mesh)
    if len(triangle_cells)!=0:
        print("write triangle_mesh")
        triangle_mesh = meshio.Mesh(points=msh.points, cells={"triangle": triangle_cells}, cell_data={"name_to_read": [triangle_data]})
        meshio.write(outputfolder + "/triangle.xdmf", triangle_mesh)
    if len(line_cells)!=0:
        print("write line_mesh")
        line_mesh = meshio.Mesh(points=msh.points, cells=[("line", line_cells)], cell_data={"name_to_read": [line_data]})
        meshio.xdmf.write(outputfolder + "/line.xdmf", line_mesh)
    if len(point_cells)!=0:
        print("write point_mesh")
        point_mesh = meshio.Mesh(points=msh.points, cells=[("vertex", point_cells)], cell_data={"name_to_read": [point_data]})
        meshio.xdmf.write(outputfolder + "/point.xdmf", point_mesh)
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
    try:
        xdmf_files = []
        if os.path.isfile(inputdirectory+"/tetra.xdmf"):
            mesh = Mesh()
            with XDMFFile(inputdirectory + "/tetra.xdmf") as infile:
                infile.read(mesh)    

            tetra_mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim())
            with XDMFFile(inputdirectory + "/tetra.xdmf") as infile:
                infile.read(tetra_mvc, "name_to_read")
            tetra = MeshFunction("size_t", mesh, tetra_mvc)
            xdmf_files.append(tetra)

            triangle_mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim())
            with XDMFFile(inputdirectory + "/triangle.xdmf") as infile:
                infile.read(triangle_mvc, "name_to_read")
            triangle = MeshFunction("size_t", mesh, triangle_mvc)
            xdmf_files.append(triangle)

            if os.path.isfile(inputdirectory+"/line.xdmf"):
                line_mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim())
                with XDMFFile(inputdirectory + "/line.xdmf") as infile:
                    infile.read(line_mvc, "name_to_read")
                line = MeshFunction("size_t", mesh, line_mvc)
                xdmf_files.append(line)

            if os.path.isfile(inputdirectory + "point.xdmf"):
                point_mvc = MeshValueCollection("size_t", mesh)
                with XDMFFile(inputdirectory + "/point.xdmf") as infile:
                    infile.read(point_mvc, "name_to_read")
                point = MeshFunction("size_t", mesh, point_mvc)
                xdmf_files.append(point)

        else:
            mesh = Mesh()
            with XDMFFile(inputdirectory + "/triangle.xdmf") as infile:
                infile.read(mesh)

            triangle_mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim())
            with XDMFFile(inputdirectory + "/triangle.xdmf") as infile:
                infile.read(triangle_mvc, "name_to_read")
            triangle = MeshFunction("size_t", mesh, triangle_mvc)
            xdmf_files.append(triangle)

            line_mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim())
            with XDMFFile(inputdirectory + "/line.xdmf") as infile:
                infile.read(line_mvc, "name_to_read")
            line = MeshFunction("size_t", mesh, line_mvc)
            xdmf_files.append(line)

            if os.path.isfile(inputdirectory+"/point.xdmf"):
                point_mvc = MeshValueCollection("size_t", mesh)
                with XDMFFile(inputdirectory + "/point.xdmf") as infile:
                    infile.read(point_mvc, "name_to_read")
                point = MeshFunction("size_t", mesh, point_mvc)
                xdmf_files.append(point)

        return xdmf_files
    except:
        print("input not working")
    pass

def getXDMF_tumor(inputdirectory):
    """
    Gathers all needed input files from a respective folder in workingdata environment and returns 
    the files in the following order: (tetra.xdmf), triangle.xdmf, lines.xmdf

    *Arguments:*
        inputdirectory: input_folder

    *Example:*
        getXDMF("Terzaghi_2d")
    """
    try:
        xdmf_files = []
        if os.path.isfile(inputdirectory+"/tetra.xdmf"):
            mesh = Mesh()
            with XDMFFile(inputdirectory + "/tetra.xdmf") as infile:
                infile.read(mesh)    

            tetra_mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim())
            with XDMFFile(inputdirectory + "/tetra.xdmf") as infile:
                infile.read(tetra_mvc, "name_to_read")
            tetra = MeshFunction("size_t", mesh, tetra_mvc)
            xdmf_files.append(tetra)

            triangle_mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim())
            with XDMFFile(inputdirectory + "/triangle.xdmf") as infile:
                infile.read(triangle_mvc, "name_to_read")
            triangle = MeshFunction("size_t", mesh, triangle_mvc)
            xdmf_files.append(triangle)

            if os.path.isfile(inputdirectory+"/line.xdmf"):
                line_mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim())
                with XDMFFile(inputdirectory + "/line.xdmf") as infile:
                    infile.read(line_mvc, "name_to_read")
                line = MeshFunction("size_t", mesh, line_mvc)
                xdmf_files.append(line)

            if os.path.isfile(inputdirectory + "point.xdmf"):
                point_mvc = MeshValueCollection("size_t", mesh)
                with XDMFFile(inputdirectory + "/point.xdmf") as infile:
                    infile.read(point_mvc, "name_to_read")
                point = MeshFunction("size_t", mesh, point_mvc)
                xdmf_files.append(point)

        else:
            mesh = Mesh()
            with XDMFFile(inputdirectory + "/triangle.xdmf") as infile:
                infile.read(mesh)

            triangle_mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim())
            with XDMFFile(inputdirectory + "/triangle.xdmf") as infile:
                infile.read(triangle_mvc, "name_to_read")
            triangle = MeshFunction("size_t", mesh, triangle_mvc)
            xdmf_files.append(triangle)

            line_mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim())
            with XDMFFile(inputdirectory + "/line.xdmf") as infile:
                infile.read(line_mvc, "name_to_read")
            line = MeshFunction("size_t", mesh, line_mvc)
            xdmf_files.append(line)

            if os.path.isfile(inputdirectory+"/point.xdmf"):
                point_mvc = MeshValueCollection("size_t", mesh)
                with XDMFFile(inputdirectory + "/point.xdmf") as infile:
                    infile.read(point_mvc, "name_to_read")
                point = MeshFunction("size_t", mesh, point_mvc)
                xdmf_files.append(point)

        return xdmf_files
    except:
        print("input not working")
    pass

def set_output_file(name: str):
    """
    Initializes xdmf file of given name. That file can be filled with multiple fields using the same mesh

    *Arguments*
        name: File path    

    *Example*
        output_file = set_output_file("solution/3d_crashtest.xdmf")
    """
    xdmf_file = XDMFFile(name+".xdmf")
    xdmf_file.rename(name, "x")
    xdmf_file.parameters["flush_output"] = True                        
    xdmf_file.parameters["functions_share_mesh"] = True    
    return xdmf_file

def write_field2output(outputfile: XDMFFile, field: Function, fieldname: str, timestep: int, id_nodes=None, mesh=None):
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
        write_field2output(xdmf_file, u, "displacement", t)
    """
    field.rename(fieldname, fieldname)
    outputfile.write(field, timestep)
    if id_nodes is not None:
        if timestep == 0:
            with open(outputfile.name() + "-" + fieldname + ".txt", "w") as myfile:
                myfile.write(str(field.value_rank())+"\t")
                for node in id_nodes:
                    myfile.write(str(node)+"\t")
                myfile.write("\n")
                myfile.write(str(timestep) + "\t")
                for node in id_nodes:
                    myfile.write(str(field(mesh.coordinates()[node]))+"\t")
                myfile.write("\n")
        else:
            with open(outputfile.name() + "-" + fieldname + ".txt", "a") as myfile:
                myfile.write(str(timestep) + "\t")
                for node in id_nodes:
                    myfile.write(str(field(mesh.coordinates()[node]).tolist())+"\t")
                myfile.write("\n")
        return [[field(mesh.coordinates()[node]), node] for node in id_nodes ]

    return True

def read_field_data(path: str):
    """

    """
    df = pd.read_csv(path, sep='\t', lineterminator='\n')
    points = df.iloc[:, 1:-1].columns.values.tolist()
    field_rank = int(df.columns.values[0])
    times = df.iloc[:, 0].values.tolist()
    name = os.path.splitext(os.path.basename(path))[0].split("-")
    if field_rank != 0:
        dim = len(ast.literal_eval(df.iloc[1, 1]))
    else:
        dim = 0

    graphs = []
    for point in points:
        if field_rank == 0:
            graph = Graph()
            graph.dim = dim
            graph.label = ",".join(name[:-1])
            graph.point = int(float(point))
            graph.x_value_list=times
            graph.field = name[-1]
            graph.y_value_list=df.loc[:, str(point)]
            graph.direction = 0
            graphs.append(graph)
        else:
            for i in range(int(dim)):
                graph = Graph()
                graph.dim = dim
                graph.label = str(i)+" "+",".join(name[:-1])
                graph.point = int(float(point))
                graph.x_value_list = times
                graph.field = name[-1]
                graph.y_value_list = [float(str(str(df.loc[:, str(point)][j]).replace("[", "").replace("]", "").split(" ")[i]).replace(",", "")) for j in range(len(df.loc[:, str(point)]))]
                graph.direction = i
                graphs.append(graph)

    return graphs

def get_data_from_txt_files(fileDir: str):
    """
    Reads out given directory for tab spaced data files
    """
    fileExt = r".txt"
    files = [os.path.join(fileDir, _) for _ in os.listdir(fileDir) if _.endswith(fileExt)]
    return [item for sublist in [read_field_data(file) for file in files] for item in sublist]
