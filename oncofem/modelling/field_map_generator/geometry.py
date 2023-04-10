"""
# **************************************************************************#
#                                                                           #
# === Geometry =============================================================#
#                                                                           #
# **************************************************************************#
# Definition of Geometry Class and severall academic examples
#
# Co-author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
#
# --------------------------------------------------------------------------#
"""

import pygmsh
import dolfin as df
import mshr
from oncofem.helper.general import run_shell_command
from oncofem.helper.io import msh2xdmf, getXDMF

# **************************************************************************#
#      Classes                                                              #
# **************************************************************************#
# Definition of Classes

# --------------------------------------------------------------------------#
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

# **************************************************************************#
#      Functions                                                            #
# **************************************************************************#
# Definition of Functions

# --------------------------------------------------------------------------#
def gen_gmsh_file(file: str, type="msh"):
    if not file.endswith("."+type):
        file += "."+type
    return file

# Geometries
def create_3D_Cube(ele_size: int, output: str):
    output = gen_gmsh_file(output)
    # generate geometry
    with pygmsh.occ.Geometry() as ge:
        p1 = ge.add_point([0.0, 0.0, 0.0])
        p2 = ge.add_point([1.0, 0.0, 0.0])
        p3 = ge.add_point([1.0, 1.0, 0.0])
        p4 = ge.add_point([0.0, 1.0, 0.0])
        p5 = ge.add_point([0.0, 0.0, 1.0])
        p6 = ge.add_point([1.0, 0.0, 1.0])
        p7 = ge.add_point([1.0, 1.0, 1.0])
        p8 = ge.add_point([0.0, 1.0, 1.0])
        l1 = ge.add_line(p1, p2)
        l2 = ge.add_line(p2, p3)
        l3 = ge.add_line(p3, p4)
        l4 = ge.add_line(p4, p1)
        l5 = ge.add_line(p1, p5)
        l6 = ge.add_line(p2, p6)
        l7 = ge.add_line(p3, p7)
        l8 = ge.add_line(p4, p8)
        l9 = ge.add_line(p5, p6)
        l10 = ge.add_line(p6, p7)
        l11 = ge.add_line(p7, p8)
        l12 = ge.add_line(p8, p5)
        lines = [l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12]
        ll = ge.add_curve_loop([l1, l2, l3, l4])
        pl1 = ge.add_plane_surface(ll)
        ll = ge.add_curve_loop([l9, l10, l11, l12])
        pl2 = ge.add_plane_surface(ll)
        ll = ge.add_curve_loop([l1, l6, -l9, -l5])
        pl3 = ge.add_plane_surface(ll)
        ll = ge.add_curve_loop([l3, l8, -l11, -l7])
        pl4 = ge.add_plane_surface(ll)
        ll = ge.add_curve_loop([l2, l7, -l10, -l6])
        pl5 = ge.add_plane_surface(ll)
        ll = ge.add_curve_loop([l4, l5, -l12, -l8])
        pl6 = ge.add_plane_surface(ll)

        vol = ge.add_surface_loop([pl1,pl2,pl3,pl4,pl5,pl6])
        v1 = ge.add_volume(vol)

        for line in lines:
            ge.set_transfinite_curve(line, int(ele_size), "Progression", 1.0)

        ge.set_transfinite_surface(pl1, "Alternated", [p1, p2, p3, p4])
        ge.set_transfinite_surface(pl2, "Alternated", [p5, p6, p7, p8])
        ge.set_transfinite_surface(pl3, "Alternated", [p1, p2, p6, p5])

        ge.add_physical(pl1, "1")
        ge.add_physical(pl2, "2")
        ge.add_physical(pl3, "3")
        ge.add_physical(pl4, "4")
        ge.add_physical(pl5, "5")
        ge.add_physical(pl6, "6")
        ge.add_physical(v1, "7")
        ge.generate_mesh(3, verbose=False)
        pygmsh.write(output)
        return True

#def create_2D_QuarterCircle(ele_size: int, output: str):
#    output = gen_msh_file(output)
#    with pygmsh.occ.Geometry() as ge:
#        p1 = ge.add_point([0.0, 0.0])
#        p2 = ge.add_point([1.0, 0.0])
#        p3 = ge.add_point([0.0, 1.0])
#        l1 = ge.add_circle_arc(p2, p1, p3)
#        l2 = ge.add_line(p3, p1)
#        l3 = ge.add_line(p1, p2)
#        ll = ge.add_curve_loop([l1, l2, l3])
#        pl = ge.add_plane_surface(ll)
#        ge.set_transfinite_curve(l1, int(ele_size * 1), "Progression", 1.0)
#        ge.set_transfinite_curve(l2, int(ele_size * 2), "Progression", 1.0)
#        ge.set_transfinite_curve(l3, int(ele_size * 2), "Progression", 1.0)
#        ge.set_transfinite_surface(pl, "Alternated", [p1, p2, p3])
#        ge.add_physical(l1, "1")
#        ge.add_physical(l2, "2")
#        ge.add_physical(l3, "3")
#        ge.add_physical(pl, "4")
#        ge.generate_mesh(2, verbose=False)
#
#        pygmsh.write(output)
#        return True

def create_2D_QuarterRing(ele_size: int, r_i: float, r_a: float, output: str):
    output = gen_gmsh_file(output)
    with pygmsh.occ.Geometry() as ge:
        p1 = ge.add_point([0.0, 0.0])
        p2 = ge.add_point([r_i, 0.0])
        p3 = ge.add_point([0.0, r_i])
        p4 = ge.add_point([r_a, 0.0])
        p5 = ge.add_point([0.0, r_a])
        l1 = ge.add_circle_arc(p2, p1, p3)
        l2 = ge.add_circle_arc(p4, p1, p5)
        l3 = ge.add_line(p2, p4)
        l4 = ge.add_line(p3, p5)
        ll = ge.add_curve_loop([l1, l4, -l2, -l3])
        pl = ge.add_plane_surface(ll)
        ge.set_transfinite_curve(l1, int(ele_size * 1), "Progression", 1.0)
        ge.set_transfinite_curve(l3, int(ele_size * 0.5), "Progression", 1.0)
        ge.set_transfinite_curve(l4, int(ele_size * 0.5), "Progression", 1.0)
        ge.set_transfinite_curve(l2, int(ele_size * 1 * r_a / r_i), "Progression", 1.0)
        ge.add_physical(l1, "1")
        ge.add_physical(l2, "2")
        ge.add_physical(l3, "3")
        ge.add_physical(l4, "4")
        ge.add_physical(pl, "5")
        ge.generate_mesh(2, verbose=False)

        pygmsh.write(output)
        return True

def create_2D_QuarterDart(ele_size: int, r_i: float, r_g: float, r_a: float, output: str):
    output = gen_gmsh_file(output)
    with pygmsh.occ.Geometry() as ge:
        p1 = ge.add_point([0.0, 0.0])
        p2 = ge.add_point([r_i, 0.0])
        p3 = ge.add_point([0.0, r_i])
        p4 = ge.add_point([r_g, 0.0])
        p5 = ge.add_point([0.0, r_g])
        p6 = ge.add_point([r_a, 0.0])
        p7 = ge.add_point([0.0, r_a])
        l1 = ge.add_circle_arc(p2, p1, p3)
        l2 = ge.add_circle_arc(p4, p1, p5)
        l3 = ge.add_circle_arc(p6, p1, p7)
        l4 = ge.add_line(p2, p4)
        l5 = ge.add_line(p4, p6)
        l6 = ge.add_line(p3, p5)
        l7 = ge.add_line(p5, p7)
        ll = ge.add_curve_loop([l1, l6, -l2, -l4])
        pl1 = ge.add_plane_surface(ll)
        ll = ge.add_curve_loop([l2, l7, -l3, -l5])
        pl2 = ge.add_plane_surface(ll)
        ge.set_transfinite_curve(l1, int(ele_size * 1), "Progression", 1.0)
        ge.set_transfinite_curve(l2, int(ele_size * 1), "Progression", 1.0)
        ge.set_transfinite_curve(l3, int(ele_size * 0.888 * r_a / r_i), "Progression", 1.0)
        ge.set_transfinite_curve(l4, int(ele_size * 0.19), "Progression", 1.0)
        ge.set_transfinite_curve(l5, int(ele_size * 0.19), "Progression", 1.0)
        ge.set_transfinite_curve(l6, int(ele_size * 0.19), "Progression", 1.0)
        ge.set_transfinite_curve(l7, int(ele_size * 0.19), "Progression", 1.0)
        ge.add_physical(l1, "1")
        ge.add_physical(l3, "2")
        ge.add_physical([l4,l5], "3")
        ge.add_physical([l6, l7], "4")
        ge.add_physical(pl1, "5")
        ge.add_physical(pl2, "6")
        ge.generate_mesh(2, verbose=False)

        pygmsh.write(output)
        return True

def create_2D_circle_in_square(ele_size: int, r: float, l: float, output: str):
    output = gen_gmsh_file(output)
    with pygmsh.occ.Geometry() as ge:
        p1 = ge.add_point([0.0, 0.0])
        p2 = ge.add_point([l, l])
        p3 = ge.add_point([-l, l])
        p4 = ge.add_point([-l, -l])
        p5 = ge.add_point([l, -l])
        p6 = ge.add_point([0, r])
        p7 = ge.add_point([0, -r])
        l1 = ge.add_circle_arc(p6, p1, p7)
        l2 = ge.add_circle_arc(p7, p1, p6)
        l3 = ge.add_line(p2, p3)
        l4 = ge.add_line(p3, p4)
        l5 = ge.add_line(p4, p5)
        l6 = ge.add_line(p5, p2)
        ll = ge.add_curve_loop([l1, l2])
        pl1 = ge.add_plane_surface(ll)
        ll = ge.add_curve_loop([l3, l4, l5, l6])
        pl2 = ge.add_plane_surface(ll)
        ge.set_transfinite_curve(l1, int(ele_size * 1), "Progression", 1.0)
        ge.set_transfinite_curve(l2, int(ele_size * 1), "Progression", 1.0)
        ge.set_transfinite_curve(l3, int(ele_size * 1), "Progression", 1.0)
        ge.set_transfinite_curve(l4, int(ele_size * 1), "Progression", 1.0)
        ge.set_transfinite_curve(l5, int(ele_size * 1), "Progression", 1.0)
        ge.set_transfinite_curve(l6, int(ele_size * 1), "Progression", 1.0)
        ge.add_physical(l4, "1") # left
        ge.add_physical(l5, "2") # bottom
        ge.add_physical(l3, "3")  # above
        ge.add_physical(l6, "4")  # right
        ge.add_physical(pl1, "5")
        ge.add_physical(pl2, "6")
        ge.generate_mesh(2, verbose=False)

        pygmsh.write(output)
        return True

def create_2D_quarter_circle_in_square(ele_size: int, r: float, l: float, output: str):
    output = gen_gmsh_file(output)
    with pygmsh.occ.Geometry() as ge:
        p1 = ge.add_point([0.0, 0.0])
        p2 = ge.add_point([r, 0.0])
        p3 = ge.add_point([0.0, r])
        p4 = ge.add_point([l, 0.0])
        p5 = ge.add_point([0.0, l])
        p6 = ge.add_point([l, l])
        l1 = ge.add_circle_arc(p2, p1, p3)
        l2 = ge.add_line(p1, p2)
        l3 = ge.add_line(p2, p4)
        l4 = ge.add_line(p4, p6)
        l5 = ge.add_line(p6, p5)
        l6 = ge.add_line(p5, p3)
        l7 = ge.add_line(p3, p1)
        ll = ge.add_curve_loop([l2, l1, l7])
        pl1 = ge.add_plane_surface(ll)
        ll = ge.add_curve_loop([l3, l4, l5, l6, -l1])
        pl2 = ge.add_plane_surface(ll)
        ge.set_transfinite_curve(l1, int(ele_size * 1.15), "Progression", 1.0)
        ge.set_transfinite_curve(l2, int(ele_size * 1), "Progression", 1.0)
        ge.set_transfinite_curve(l3, int(ele_size * 1), "Progression", 1.0)
        ge.set_transfinite_curve(l4, int(ele_size * 1), "Progression", 1.0)
        ge.set_transfinite_curve(l5, int(ele_size * 1), "Progression", 1.0)
        ge.set_transfinite_curve(l6, int(ele_size * 1), "Progression", 1.0)
        ge.set_transfinite_curve(l7, int(ele_size * 1), "Progression", 1.0)
        ge.add_physical(l4, "1")
        ge.add_physical(l5, "2")
        ge.add_physical([l2,l3], "3") #x
        ge.add_physical([l6, l7], "4") #y
        ge.add_physical(pl1, "5")
        ge.add_physical(pl2, "6")
        ge.generate_mesh(2, verbose=False)

        pygmsh.write(output)
        return True

def create_2D_quarter_circle_in_rectangle(ele_size: int, r: float, l_1: float, l_2: float, output: str):
    output = gen_gmsh_file(output)
    with pygmsh.occ.Geometry() as ge:
        p1 = ge.add_point([0.0, 0.0])
        p2 = ge.add_point([r, 0.0])
        p3 = ge.add_point([0.0, r])
        p4 = ge.add_point([l_1, 0.0])
        p5 = ge.add_point([0.0, l_2])
        p6 = ge.add_point([l_1, l_2])
        l1 = ge.add_circle_arc(p2, p1, p3)
        l2 = ge.add_line(p1, p2)
        l3 = ge.add_line(p2, p4)
        l4 = ge.add_line(p4, p6)
        l5 = ge.add_line(p6, p5)
        l6 = ge.add_line(p5, p3)
        l7 = ge.add_line(p3, p1)
        ll = ge.add_curve_loop([l2, l1, l7])
        pl1 = ge.add_plane_surface(ll)
        ll = ge.add_curve_loop([l3, l4, l5, l6, -l1])
        pl2 = ge.add_plane_surface(ll)
        ge.set_transfinite_curve(l1, int(ele_size * 1.15), "Progression", 1.0)
        ge.set_transfinite_curve(l2, int(ele_size * 1), "Progression", 1.0)
        ge.set_transfinite_curve(l3, int(ele_size * 1), "Progression", 1.0)
        ge.set_transfinite_curve(l4, int(ele_size * 1), "Progression", 1.0)
        ge.set_transfinite_curve(l5, int(ele_size * 1), "Progression", 1.0)
        ge.set_transfinite_curve(l6, int(ele_size * 1), "Progression", 1.0)
        ge.set_transfinite_curve(l7, int(ele_size * 1), "Progression", 1.0)
        ge.add_physical(l4, "1")
        ge.add_physical(l5, "2")
        ge.add_physical([l2,l3], "3") #x
        ge.add_physical([l6, l7], "4") #y
        ge.add_physical(pl1, "5")
        ge.add_physical(pl2, "6")
        ge.generate_mesh(2, verbose=False)

        pygmsh.write(output)
        return True

def create_3D_quarter_tube(ele_size: int, r_i: float, r_a: float, l_1: float, l_2: float, output: str):
    output = gen_gmsh_file(output)
    with pygmsh.occ.Geometry() as ge:
        p1 = ge.add_point([0.0, 0.0, 0.0])
        p2 = ge.add_point([r_i, 0.0, 0.0])
        p3 = ge.add_point([0.0, r_i, 0.0])
        p4 = ge.add_point([r_a, 0.0, 0.0])
        p5 = ge.add_point([0.0, r_a, 0.0])
        p6 = ge.add_point([0.0, 0.0, l_1])
        p7 = ge.add_point([r_i, 0.0, l_1])
        p8 = ge.add_point([0.0, r_i, l_1])
        p9 = ge.add_point([r_a, 0.0, l_1])
        p10 = ge.add_point([0.0, r_a, l_1])
        p11 = ge.add_point([0.0, 0.0, -l_2])
        p12 = ge.add_point([r_i, 0.0, -l_2])
        p13 = ge.add_point([0.0, r_i, -l_2])
        p14 = ge.add_point([r_a, 0.0, -l_2])
        p15 = ge.add_point([0.0, r_a, -l_2])

        l1 = ge.add_circle_arc(p2, p1, p3)
        l2 = ge.add_circle_arc(p4, p1, p5)
        l3 = ge.add_line(p2, p4)
        l4 = ge.add_line(p3, p5)
        l5 = ge.add_circle_arc(p7, p6, p8)
        l6 = ge.add_circle_arc(p9, p6, p10)
        l7 = ge.add_line(p7, p9)
        l8 = ge.add_line(p8, p10)
        l9 = ge.add_circle_arc(p12, p11, p13)
        l10 = ge.add_circle_arc(p14, p11, p15)
        l11 = ge.add_line(p12, p14)
        l12 = ge.add_line(p13, p15)
        l13 = ge.add_line(p12, p2)
        l14 = ge.add_line(p2, p7)
        l15 = ge.add_line(p9, p4)
        l16 = ge.add_line(p4, p14)
        l17 = ge.add_line(p13, p3)
        l18 = ge.add_line(p3, p8)
        l19 = ge.add_line(p10, p5)
        l20 = ge.add_line(p5, p15)

        ll = ge.add_curve_loop([l13, l3, l16, -l11])
        pl1 = ge.add_plane_surface(ll)
        ll = ge.add_curve_loop([l14, l7, l15, -l3])
        pl2 = ge.add_plane_surface(ll)
        ll = ge.add_curve_loop([l7, l6, -l8, -l5])
        pl3 = ge.add_plane_surface(ll)
        ll = ge.add_curve_loop([l11, l10, -l12, -l9])
        pl4 = ge.add_plane_surface(ll)
        ll = ge.add_curve_loop([l17, l4, l20, -l12])
        pl5 = ge.add_plane_surface(ll)
        ll = ge.add_curve_loop([l18, l8, l19, -l4])
        pl6 = ge.add_plane_surface(ll)
        ll = ge.add_curve_loop([l13, l1, -l17, -l9])
        pl7 = ge.add_surface(ll)
        ll = ge.add_curve_loop([l14, l5, -l18, -l1])
        pl8 = ge.add_surface(ll)
        ll = ge.add_curve_loop([-l16, l2, l20, -l10])
        pl9 = ge.add_surface(ll)
        ll = ge.add_curve_loop([-l15, l6, l19, -l2])
        pl10 = ge.add_surface(ll)
        ll = ge.add_curve_loop([l3, l2, -l4, -l1])
        pl11 = ge.add_plane_surface(ll)

        pl = ge.add_surface_loop([pl2, pl3, pl6, pl11, pl8, pl10])
        vl1 = ge.add_volume(pl)
        pl = ge.add_surface_loop([pl1, pl7, pl11, pl9, pl4, pl5])
        vl2 = ge.add_volume(pl)

        ge.set_transfinite_curve(l1, int(ele_size * 1), "Progression", 1.0)
        ge.set_transfinite_curve(l2, int(ele_size * 1.5), "Progression", 1.0)
        ge.set_transfinite_curve(l3, int(ele_size * 0.2), "Progression", 1.0)
        ge.set_transfinite_curve(l4, int(ele_size * 0.2), "Progression", 1.0)
        ge.set_transfinite_curve(l5, int(ele_size * 1.0), "Progression", 1.0)
        ge.set_transfinite_curve(l6, int(ele_size * 1.5), "Progression", 1.0)
        ge.set_transfinite_curve(l7, int(ele_size * 0.2), "Progression", 1.0)
        ge.set_transfinite_curve(l8, int(ele_size * 0.2), "Progression", 1.0)
        ge.set_transfinite_curve(l9, int(ele_size * 1), "Progression", 1.0)
        ge.set_transfinite_curve(l10, int(ele_size * 1.5), "Progression", 1.0)
        ge.set_transfinite_curve(l11, int(ele_size * 0.2), "Progression", 1.0)
        ge.set_transfinite_curve(l12, int(ele_size * 0.2), "Progression", 1.0)
        ge.set_transfinite_curve(l13, int(ele_size * 1), "Progression", 1.0)
        ge.set_transfinite_curve(l14, int(ele_size * 2), "Progression", 1.0)
        ge.set_transfinite_curve(l15, int(ele_size * 2), "Progression", 1.0)
        ge.set_transfinite_curve(l16, int(ele_size * 1), "Progression", 1.0)
        ge.set_transfinite_curve(l17, int(ele_size * 1), "Progression", 1.0)
        ge.set_transfinite_curve(l18, int(ele_size * 2), "Progression", 1.0)
        ge.set_transfinite_curve(l19, int(ele_size * 2), "Progression", 1.0)
        ge.set_transfinite_curve(l20, int(ele_size * 1), "Progression", 1.0)

        ge.add_physical([pl1, pl2], "1")    # y
        ge.add_physical([pl5, pl6], "2")    # x
        ge.add_physical(pl3, "3")           # z
        ge.add_physical(vl1, "5")
        ge.add_physical(vl2, "6")
        ge.generate_mesh(3, verbose=False)

        pygmsh.write(output)
        return True

def create_2D_QuarterCircle_Tumor(ele_size: float, fac: float, radius: float, i_radius: float, layer: int, der_file: str, der_path: str, concentration: float, diff: float):
    output = gen_gmsh_file(der_file, "geo")
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

    done = run_shell_command("gmsh " + output + " -2")
    msh2xdmf(der_file, der_path)
    _, facet_function = getXDMF(der_path)
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

def create_intern_rectangle(length, height, ele_l, ele_h, type="crossed"):
    msh = df.RectangleMesh(df.Point(0.0,0.0), df.Point(length, height), ele_l, ele_h, type)
    bndry = df.MeshFunction("size_t", msh, msh.topology().dim()-1)
    for f in df.facets(msh):
        mp = f.midpoint()
        if df.near(mp[0], 0.0):  # inflow
            bndry[f] = 1
        elif df.near(mp[0], length):  # outflow
            bndry[f] = 2
        elif df.near(mp[1], 0.0):  # bottom
            bndry[f] = 3
        elif df.near(mp[1], height):  # cylinder
            bndry[f] = 4
    return msh, bndry

def create_tumour_in_quarter_circle(radius, inner_circle_rad,  n_circle, n_mesh, concentration):
    center = df.Point(0.0, 0.0)
    circle_inner = mshr.Circle(center, inner_circle_rad, n_circle)
    circle = mshr.Circle(center, radius, n_circle)
    rec1 = mshr.Rectangle(df.Point(-radius, -radius), df.Point(0.0, radius))
    rec2 = mshr.Rectangle(df.Point(0.0, -radius), df.Point(radius, 0.0))
    geo = (((circle-circle_inner)+circle_inner) - rec1) - rec2
    mesh = mshr.generate_mesh(geo, n_mesh)
    bndry = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    domain = df.MeshFunction("double", mesh, mesh.topology().dim())
    init_circle = df.AutoSubDomain(lambda x: x[0]*x[0]+x[1]*x[1] < inner_circle_rad)
    bottom = df.AutoSubDomain(lambda x: df.near(x[1], 0.0))
    left = df.AutoSubDomain(lambda x: df.near(x[0], 0.0))

    bottom.mark(bndry, 1)
    left.mark(bndry, 2)
    init_circle.mark(domain, concentration)

    return mesh, bndry, domain
