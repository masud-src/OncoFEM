"""
Beispiel 2Phaser TPM + poisson 2D
Start bei fieldmapping
kopplung 

# File of model paper calculation
#
# Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
#
# --------------------------------------------------------------------------#
"""

# Imports
import time
import os
import oncofem.struc as str
from oncofem.struc.problem import Problem
from oncofem.helper.io import set_output_file
import oncofem.modelling.base_model.glioblastoma as bm
from oncofem.modelling.bio_chem_models.resection_model import ResectionModel
from oncofem.modelling.bio_chem_models.simple_model import SimpleModel
import academic_geometries
import dolfin as df 
import numpy as np

# define study
study = str.Study("resection")
x = Problem()

# geometry
x.param.gen.title = "Start"
x.geom.dim = 2
der_file = study.der_dir + x.param.gen.title
der_path = der_file + os.sep

x.geom.mesh, x.geom.facet_function, area_conc, area_df = academic_geometries.create_2D_QuarterCircle_Tumor(0.0001, 1000.0, 1.0, 0.0006, 40, der_file, 1.15E-13, 1e-5)  # 0.01 60

################################################################################################################
# BASE MODEL
# general info
x.param.gen.flag_defSplit = True

# time parameters
x.param.time.T_end = 29.0  # *86400
x.param.time.output_interval = 24.0/24.0  # *86400
x.param.time.dt = 3.0/24.0  # *86400

# material parameters base model
x.param.mat.rhoShR = 1190.0
x.param.mat.rhoStR = 1190.0  # muss größer sein als Sh
x.param.mat.rhoSnR = 1190.0
x.param.mat.rhoFR = 1993.3
x.param.mat.gammaFR = 1.0
x.param.mat.molFt = 2.018E13
x.param.mat.R = 8.31446261815324
x.param.mat.Theta = 37.0

# spatial varying material parameters
x.param.mat.lambdaSh = 3312.0
x.param.mat.lambdaSt = 3312.0
x.param.mat.lambdaSn = 3312.0
x.param.mat.muSh = 662.0
x.param.mat.muSt = 662.0
x.param.mat.muSn = 662.0
x.param.mat.kF = 5E-13
x.param.mat.DFt = 1.5E-13 * 86400

# FEM Paramereters
x.param.fem.solver_type = "lu"
x.param.fem.maxIter = 20
x.param.fem.rel = 1E-7
x.param.fem.abs = 1E-8
################################################################################################################

################################################################################################################
# ADDITIONALS
# material parameters
molFn = 0.18
DFn = 6.6E-10 * 86400
x.param.add.prim_vars = ["cFn"]
x.param.add.ele_types = ["CG"]
x.param.add.ele_orders = [1] 
x.param.add.tensor_orders = [0]
x.param.add.molFkappa = [molFn]
x.param.add.DFkappa = [DFn]
################################################################################################################
print("Start calculation")
df.set_log_level(30)
start = time.time()  # start time
model = bm.Glioblastoma()
file = set_output_file(study.sol_dir + x.param.gen.title + "/TPM")
x.param.gen.output_file = file
model.set_param(x)
model.set_function_spaces()

################################################################################################################
# initial conditions
x.param.init.uS_0S = [0.0, 0.0]
x.param.init.p_0S = 0.0
x.param.init.nSh_0S = 0.4 
x.param.init.nSt_0S = 0.0  # fmg.read_mapped_xdmf(fmg.mapped_solid_tumor_file)
x.param.init.nSn_0S = 0.0  # fmg.read_mapped_xdmf(fmg.mapped_necrotic_file)
V = df.FunctionSpace(x.geom.mesh, "CG", 2)
field = df.Expression(("ct0*exp(-a*(pow((x[0]-x_source),2)+pow((x[1]-y_source),2)))"), degree=2, ct0=6.15e-1, a=100, x_source=0.0, y_source=0.0)  # 1.15e-1
area_cFt = df.interpolate(field, model.CG1_sca)
x.param.init.cFt_0S = area_cFt  # field #fmg.read_mapped_xdmf(init_cFt)
cFn_0S = 1.0
cFa_0S = 0.0
x.param.add.cFkappa_0S = [cFn_0S]

################################################################################################################
# Bio chemical set up
bio_model = SimpleModel()
bio_model.set_prim_vars(model.ansatz_functions)
bio_model.set_param(x)
bio_model.flag_proliferation = True
bio_model.flag_metabolism = True
bio_model.flag_necrosis = True
bio_model.nSt_thres_lin_ms = 5e-5
bio_model.fac_nSt_lin_ms = 1e-1
bio_model.nu_Sh_necrosis = 1e-15 * 86400
bio_model.nu_St_necrosis = 1E-15 * 86400
bio_model.nu_Ft_necrosis = 0.0 * 86400
bio_model.cFn_min_necrosis = 0.85
bio_model.nSt_max = 0.5
bio_model.cFt_max = 9.828212E-1
bio_model.cFn_min_growth = 0.35
bio_model.nu_In_basal = 8.64e-28
bio_model.nu_Ft_proliferation = 0.0864
bio_model.nu_St_proliferation = 0.35856e-3  # 0.35856
bio_model.f_proli = 8.64e-5
prod_list = bio_model.return_prod_terms()
model.set_bio_chem_models(prod_list)
################################################################################################################
# Boundary conditions
# u (x,y,z), p, nSh, nSt, nSn, cFt, cFn, cFa
bc_u_0 = df.DirichletBC(model.function_space.sub(0).sub(0), 0.0, x.geom.facet_function, 3)
bc_u_1 = df.DirichletBC(model.function_space.sub(0).sub(1), 0.0, x.geom.facet_function, 2)
bc_p_0 = df.DirichletBC(model.function_space.sub(1), 0.0, x.geom.facet_function, 4)
bc_cFn_1 = df.DirichletBC(model.function_space.sub(6), 1.0, x.geom.facet_function, 4)
################################################################################################################

model.set_boundaries([bc_u_0, bc_u_1, bc_p_0, bc_cFn_1], None)
model.set_heterogenities()
model.set_weak_form()
model.set_solver()
model.set_initial_conditions(x.param.init, x.param.add)
model.solve() 

def create_cutted_mesh(mesh: df.Mesh, field: df.Function, tol: float, dim: int):
    """
    t.b.d.
    """
    # get inices of values of field higher tolerance
    values = field.compute_vertex_values()
    idx = []
    for i, val in enumerate(values):
        if val > tol:
            idx.append(i)
    idx.sort(reverse=True)

    # get remaining nodes
    old_cells = mesh.cells()
    old_coors = mesh.coordinates()
    nodes = old_coors.tolist()
    for id in idx:
        nodes.pop(id)

    if dim == 2:
        for i, node in enumerate(nodes):
            nodes[i] = [node[0], node[1]]

    # get remaining elements
    elements = old_cells
    del_index = []

    for i, element in enumerate(old_cells):
        for id in idx:
            if dim == 2:
                if element[0] == id or element[1] == id or element[2] == id:
                    del_index.append(i)
                    break
            if dim == 3:
                if element[0] == id or element[1] == id or element[2] == id or element[3] == id:
                    del_index.append(i)
                    break

    del_index = np.unique(del_index)
    elements = np.delete(elements, del_index, axis=0)

    # correct indices
    for id in idx:
        for i, element in enumerate(elements):
            for j in range(3):
                if element[j] > id:
                    elements[i][j] -= 1

    # create new mesh
    new_mesh = df.Mesh()
    editor = df.MeshEditor()
    editor.open(new_mesh, mesh.ufl_cell().cellname(), mesh.geometry().dim(), mesh.topology().dim())
    editor.init_vertices(len(nodes))
    editor.init_cells(len(elements))

    for i, node in enumerate(nodes):
        editor.add_vertex(i, node)

    for i, element in enumerate(elements):
        editor.add_cell(i, element)

    editor.close()
    return new_mesh, idx

def map_old_2_new(field_old: df.Function, V_new: df.FunctionSpace, vertex_id: list):
    vals = field_old.compute_vertex_values().tolist()
    for id in vertex_id:
        vals.pop(id)
    field_new = df.Function(V_new)
    for i, val in enumerate(vals):
        field_new.vector()[df.vertex_to_dof_map(V)[i]] = val
    return field_new

def write_checkpoint(field, name):
    xdmf_file = df.XDMFFile(study.sol_dir + "inter" + os.sep + name + ".xdmf")
    xdmf_file.write(field)
    xdmf_file.write_checkpoint(field, name, 0, df.XDMFFile.Encoding.HDF5, False)
    xdmf_file.close()

def read_checkpoint(functionspace, name):
    f_in = df.XDMFFile(name)
    f = df.Function(functionspace)
    f_in.read_checkpoint(f, "f", 0)
    return f

new_mesh, idx = create_cutted_mesh(model.mesh, model.sol.sub(3), 1.0e-5, 2)
V = df.FunctionSpace(new_mesh, "CG", 1)
V2 = df.VectorFunctionSpace(new_mesh, "CG", 2)

u = model.sol.sub(0)
u_new = df.interpolate(u, V2)
write_checkpoint(u_new, "u")
write_checkpoint(map_old_2_new(model.sol.sub(1), V, idx), "p")
write_checkpoint(map_old_2_new(model.sol.sub(2), V, idx), "nSh")
write_checkpoint(map_old_2_new(model.sol.sub(3), V, idx), "nSt")
write_checkpoint(map_old_2_new(model.sol.sub(4), V, idx), "nSn")
write_checkpoint(map_old_2_new(model.sol.sub(5), V, idx), "cFt")
write_checkpoint(map_old_2_new(model.sol.sub(6), V, idx), "cFn")

#f_in = df.XDMFFile("test.xdmf")
#f1 = df.Function(V)
#f_in.read_checkpoint(f1, "f", 0)

# Create a MeshFunction for the boundary
boundary_markers = df.MeshFunction("size_t", new_mesh, new_mesh.topology().dim() - 1)

# Mark different boundaries with different values
left_boundary = df.CompiledSubDomain("near(x[0], 0)")
bottom_boundary = df.CompiledSubDomain("near(x[1], 0)")
r = 1.0  # radius
expr = "x[0] * x[0] + x[1] * x[1] >= 0.9".format(r)
outer_circle = df.CompiledSubDomain(expr)
expr = "x[0] * x[0] + x[1] * x[1] <= 0.01".format(r)
inner_circle = df.CompiledSubDomain(expr)
boundary_markers.set_all(0)  # Initialize all markers to 0
left_boundary.mark(boundary_markers, 3)
outer_circle.mark(boundary_markers, 4)
inner_circle.mark(boundary_markers, 1)
bottom_boundary.mark(boundary_markers, 2)
# Save the boundary markers to a file
with df.XDMFFile("boundary_markers.xdmf") as file:
    file.write(boundary_markers)

p = x
p.param.gen.title = "2D_resected_area"
p.geom.dim = 2
der_file = study.der_dir + x.param.gen.title
der_path = der_file + os.sep

p.geom.mesh = new_mesh
p.geom.facet_function = boundary_markers

################################################################################################################
# BASE MODEL
# general info
p.param.gen.flag_defSplit = True

# time parameters
p.param.time.T_end = 29.0  # *86400
p.param.time.output_interval = 24.0/24.0  # *86400
p.param.time.dt = 3.0/24.0  # *86400

# FEM Paramereters
p.param.fem.solver_type = "lu"
p.param.fem.maxIter = 20
p.param.fem.rel = 1E-7
p.param.fem.abs = 1E-8
################################################################################################################

################################################################################################################
# ADDITIONALS
# material parameters
molFn = 0.18
DFn = 6.6E-10 * 86400
molFd = 93
DFd = 1E-11 * 86400
p.param.add.prim_vars = ["cFn", "cFd"]
p.param.add.ele_types = ["CG", "CG"]
p.param.add.ele_orders = [1, 1] 
p.param.add.tensor_orders = [0, 0]
p.param.add.molFkappa = [molFn, molFd]
p.param.add.DFkappa = [DFn, DFd]
################################################################################################################
print("Start calculation")
df.set_log_level(30)
start = time.time()  # start time
new_model = bm.Glioblastoma()
file = set_output_file(study.sol_dir + x.param.gen.title + "/TPM")
p.param.gen.output_file = file
new_model.set_param(p)
new_model.set_function_spaces()

################################################################################################################
# initial conditions
p.param.init.uS_0S = u_new
p.param.init.p_0S = map_old_2_new(model.sol.sub(1), V, idx)
p.param.init.nSh_0S = map_old_2_new(model.sol.sub(2), V, idx)
p.param.init.nSt_0S = 0.0  # fmg.read_mapped_xdmf(fmg.mapped_solid_tumor_file)
p.param.init.nSn_0S = 0.0  # fmg.read_mapped_xdmf(fmg.mapped_necrotic_file)
p.param.init.cFt_0S = map_old_2_new(model.sol.sub(5), V, idx)  # field #fmg.read_mapped_xdmf(init_cFt)
cFn_0S = 1.0
cFa_0S = 0.0
p.param.add.cFkappa_0S = [cFn_0S, cFa_0S]

################################################################################################################
# Bio chemical set up
bio_model = ResectionModel()
bio_model.set_prim_vars(new_model.ansatz_functions)
bio_model.set_param(p)
bio_model.flag_proliferation = True
bio_model.flag_metabolism = True
bio_model.flag_necrosis = True
bio_model.flag_drug = True
bio_model.nSt_thres_lin_ms = 5e-5
bio_model.fac_nSt_lin_ms = 1e-1
bio_model.nu_Sh_necrosis = 1e-15 * 86400
bio_model.nu_St_necrosis = 1E-15 * 86400
bio_model.nu_Ft_necrosis = 0.0 * 86400
bio_model.cFn_min_necrosis = 0.85
bio_model.nSt_max = 0.5
bio_model.cFt_max = 9.828212E-1
bio_model.cFn_min_growth = 0.35
bio_model.nu_In_basal = 8.64e-28
bio_model.nu_Ft_proliferation = 0.0864
bio_model.nu_St_proliferation = 0.35856e-3  # 0.35856
bio_model.f_proli = 8.64e-5
prod_list = bio_model.return_prod_terms()
new_model.set_bio_chem_models(prod_list)
################################################################################################################
# Boundary conditions
# u (x,y,z), p, nSh, nSt, nSn, cFt, cFn, cFa
bc_u_0 = df.DirichletBC(new_model.function_space.sub(0).sub(0), 0.0, p.geom.facet_function, 3)
bc_u_1 = df.DirichletBC(new_model.function_space.sub(0).sub(1), 0.0, p.geom.facet_function, 2)
bc_p_0 = df.DirichletBC(new_model.function_space.sub(1), 0.0, p.geom.facet_function, 4)
bc_cFd_1 = df.DirichletBC(new_model.function_space.sub(7), 1.16773E-7, p.geom.facet_function, 1)
################################################################################################################

new_model.set_boundaries([bc_u_0, bc_u_1, bc_p_0, bc_cFd_1], None)
new_model.set_heterogenities()
new_model.set_weak_form()
new_model.set_solver()
new_model.set_initial_conditions(p.param.init, p.param.add)
new_model.solve() 


