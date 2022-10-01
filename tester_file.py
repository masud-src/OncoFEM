from dolfin import *
import numpy as np

class left_side(SubDomain):
    def inside(self, x, on_boundary):
        if x_bounds is None:
            x_max = np.max(mesh.coordinates()[:, 0])
            x_min = np.min(mesh.coordinates()[:, 0])
            x_b = (x_min, x_max)
        else:
            x_b = x_bounds
        if y_bounds is None:
            y_max = np.max(mesh.coordinates()[:, 0])
            y_min = np.min(mesh.coordinates()[:, 0])
            y_b = (y_min, y_max)
        else:
            y_b = y_bounds
        cond1 = between(x[0], x_b)    
        cond2 = between(x[1], y_b)
        in_bounding_box = cond1 and cond2
        return in_bounding_box and on_boundary


mesh = UnitSquareMesh(10, 10)
x_bounds = (0, 0.5)
y_bounds = None
markers = MeshFunction("size_t", mesh,  mesh.topology().dim()-1, 0)

left_side().mark(markers, 1)

File("markers.pvd") << markers
