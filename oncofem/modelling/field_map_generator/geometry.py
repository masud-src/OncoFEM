"""
Definition of Geometry Class and several academic examples

Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""

import dolfin as df
import oncofem.helper.general as gn
import oncofem.helper.io as io

#############################################################################
# Geometry class
class Geometry:
    """
    defines the geometry of a problem

    *Parameters:*
        mesh: generated mesh from xdmf format
        element: finite element definition
        facet_function: geometrical faces 
        domain: geometrical domains
        function_space: function_space of primary variables
        ansatz_function: List of ansatz functions
        test_functions: List of test functions
        dx: Integration measure of the body, Can be listed
        faces: List of curves or faces
        d_bound: List of Dirichlet boundaries
        n_bound: List of Neumann boundaries

    *Example*
    geom = geometry()
    geom.mesh = RectangleMesh(P1,P2, eleX, eleY)
    """

    def __init__(self):
        self.mesh = None
        self.dim = None
        self.element = None
        self.facet_function = None
        self.domain = None
        self.domain_list = None
        self.function_space = None
        self.ansatz_function = None
        self.test_function = None
        self.dx = None
        self.faces = None
        self.d_bound = None
        self.n_bound = None

#############################################################################
# Academic examples of geometries
def create_2D_QuarterCircle(ele_size: float, fac: float, radius: float, layer: int, der_file: str, der_path: str):
    output = gn.add_file_appendix(der_file, "geo")
    with open(output, 'w') as f:
        f.write("SetFactory(\"OpenCASCADE\");\n")
        f.write("Point(1) = {0, 0, 0, "+str(ele_size)+"};\n")
        f.write("Point(2) = {"+str(radius)+", 0, 0, "+str(ele_size*fac)+"};\n")
        f.write("Line(1) = {1, 2};\n")
        f.write("Extrude {{0, 0, 1}, {0, 0, 0}, Pi/2} {\n")
        f.write("  Curve{1}; Layers{"+str(layer)+"};\n")
        f.write("}\n")
        f.write("Physical Surface(\"4\") = {1};\n")
        f.write("Physical Curve(\"1\") = {1};\n")
        f.write("Physical Curve(\"2\") = {3};\n")
        f.write("Physical Curve(\"3\") = {2};\n")

    done = gn.run_shell_command("gmsh " + output + " -2")
    io.msh2xdmf(der_file, der_path)
    _, facet_function = io.getXDMF(der_path)
    mesh = facet_function.mesh()
    bndry = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    outer_circle = df.AutoSubDomain(lambda x: x[0] * x[0] + x[1] * x[1] >= radius-df.DOLFIN_EPS)
    bottom = df.AutoSubDomain(lambda x: df.near(x[1], 0.0))
    left = df.AutoSubDomain(lambda x: df.near(x[0], 0.0))

    bottom.mark(bndry, 1)
    left.mark(bndry, 2)
    outer_circle.mark(bndry, 3)

    return mesh, bndry

def create_2D_QuarterCircle_Tumor(ele_size: float, fac: float, radius: float, i_radius: float, layer: int, der_file: str, der_path: str, concentration: float, diff: float):
    output = gn.add_file_appendix(der_file, "geo")
    with open(output, 'w') as f:
        f.write("SetFactory(\"OpenCASCADE\");\n")
        f.write("Point(1) = {0, 0, 0, "+str(ele_size)+"};\n")
        f.write("Point(2) = {"+str(radius)+", 0, 0, "+str(ele_size*fac)+"};\n")
        f.write("Line(1) = {1, 2};\n")
        f.write("Extrude {{0, 0, 1}, {0, 0, 0}, Pi/2} {\n")
        f.write("  Curve{1}; Layers{"+str(layer)+"};\n")
        f.write("}\n")
        f.write("Physical Surface(\"4\") = {1};\n")
        f.write("Physical Curve(\"1\") = {1};\n")
        f.write("Physical Curve(\"2\") = {3};\n")
        f.write("Physical Curve(\"3\") = {2};\n")

    done = gn.run_shell_command("gmsh " + output + " -2")
    io.msh2xdmf(der_file, der_path)
    _, facet_function = io.getXDMF(der_path)
    mesh = facet_function.mesh()
    bndry = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    domain = df.MeshFunction("double", mesh, mesh.topology().dim())
    dFa = df.MeshFunction("double", mesh, mesh.topology().dim())
    init_circle = df.AutoSubDomain(lambda x: x[0] * x[0] + x[1] * x[1] < i_radius)
    df_circle = df.AutoSubDomain(lambda x: x[0] * x[0] + x[1] * x[1] <= i_radius+0.001)
    outer_circle = df.AutoSubDomain(lambda x: x[0] * x[0] + x[1] * x[1] >= radius-df.DOLFIN_EPS)
    bottom = df.AutoSubDomain(lambda x: df.near(x[1], 0.0))
    left = df.AutoSubDomain(lambda x: df.near(x[0], 0.0))

    bottom.mark(bndry, 1)
    left.mark(bndry, 2)
    init_circle.mark(domain, concentration)
    outer_circle.mark(bndry, 3)
    df_circle.mark(dFa, diff*2)

    return mesh, facet_function, domain, dFa
