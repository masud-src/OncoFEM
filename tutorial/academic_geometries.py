"""
Definition of academic geometry examples

Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""

import dolfin as df
from oncofem.helper.general import add_file_appendix, run_shell_command
from oncofem.helper.io import msh2xdmf, getXDMF
import os

def create_2D_QuarterCircle(ele_size: float, fac: float, radius: float, layer: int, der_file: str, struc_mesh=True):
    der_path = der_file + os.sep
    output = add_file_appendix(der_file, "geo")
    ele_size = ele_size * radius
    if struc_mesh:
        with open(output, 'w') as f:
            f.write("SetFactory(\"OpenCASCADE\");\n")
            f.write("Point(1) = {0, 0, 0, " + str(ele_size) + "};\n")
            f.write("Point(2) = {" + str(radius) + ", 0, 0, " + str(ele_size * fac) + "};\n")
            f.write("Line(1) = {1, 2};\n")
            f.write("Extrude {{0, 0, 1}, {0, 0, 0}, Pi/2} {\n")
            f.write("  Curve{1}; Layers{" + str(layer) + "};\n")
            f.write("}\n")
            f.write("Physical Surface(\"4\") = {1};\n")
            f.write("Physical Curve(\"1\") = {1};\n")
            f.write("Physical Curve(\"2\") = {3};\n")
            f.write("Physical Curve(\"3\") = {2};\n")
    else:
        with open(output, 'w') as f:
            f.write("SetFactory(\"OpenCASCADE\");\n")
            f.write("Point(1) = {0, 0, 0, " + str(ele_size) + "};\n")
            f.write("Point(2) = {" + str(radius) + ", 0, 0, " + str(ele_size * fac) + "};\n")
            f.write("Point(3) = {0, " + str(radius) + ", 0, " + str(ele_size * fac) + "};\n")
            f.write("Line(1) = {1, 2};\n")
            f.write("Line(2) = {3, 1};\n")
            f.write("Circle(3) = {2, 1, 3};\n")
            f.write("Curve Loop(1) = {2, 1, 3};\n")
            f.write("Surface(1) = {1};\n")
            f.write("Physical Surface(\"4\") = {1};\n")
            f.write("Physical Curve(\"1\") = {1};\n")
            f.write("Physical Curve(\"2\") = {3};\n")
            f.write("Physical Curve(\"3\") = {2};\n")
    done = run_shell_command("gmsh " + output + " -2")
    msh2xdmf(der_file, der_path, correct_gmsh=True)
    _, facet_function = getXDMF(der_path)
    mesh = facet_function.mesh()
    bndry = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    outer_circle = df.AutoSubDomain(lambda x: df.near(x[0] * x[0] + x[1] * x[1], radius*radius, eps=1e-3))
    bottom = df.AutoSubDomain(lambda x: df.near(x[1], 0.0))
    left = df.AutoSubDomain(lambda x: df.near(x[0], 0.0))

    bottom.mark(bndry, 1)
    left.mark(bndry, 2)
    outer_circle.mark(bndry, 3)

    return mesh, bndry
