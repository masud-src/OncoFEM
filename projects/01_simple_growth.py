"""
Academic test geometry tutorial

In this second tutorial, a simplified quarter circle domain shall preserve the basis for further investigations. Herein,
again the set of governing equations read

 0 = div T - hatrhoF w_F
 0 = (nS)'_S + nS div x'_S - hatnS
 0 = div x'_S + div(nF w_F) - (hatrhoS / rhoSR + hatrhoF / rhoFR)
 0 = nF (cFt_m)'_S + div(nF cFt_m w_Ft) + cFt_m (div x'_S - hatrhoS / rhoS) - hatrhoFt / MFt_m

and are solved for the primary variables of the solid displacement u_S, the solid volume fraction nS, the fluid pressure
pF and the tumor cell concentration cFt. Alternatively, to the first tutorial, the coupling to the solid body of the 
micro model will be activated and a swelling can be observed.

Notes:
A gmsh ascii file is created and meshed. This mesh is loaded and a geometry object is created. Nevertheless, again a 
study is set up, which will set up the workspace on the hard disc. There is no need for the handling of medical image 
data and therefore, the simple test case is set up by defining the problem and creating the geometry with the predefined 
function 'create_Quarter_Circle'. In the following, the problem is set up by adding various parameters. Diverging from 
the first tutorial the initial conditions are set via scalar homogeneous values and functions. The parameters of the 
growth function need to be scaled, because of the changed problem. 
"""
# Imports
import oncofem as of
import dolfin as df
########################################################################################################################
# SIMULATION
#
# Predefined function for mesh creation. The function takes the maximum element length at the centre. A factor for the
# element size at the outer corners. The radius of the quarter circle. Next paramter sets the element layers the quarter
# circle is split into. A name for the .mesh and .geo file need to be set and a boolean need to be set, which controls 
# the mesh to be either a structural or unstructural mesh.
def create_Rectangle(length: float, width: float, esize_length: float, 
                     esize_width: float, dfile: str) -> of.simulation.problem.Geometry():
    """
    creates a 2D quarter circle with three boundary conditions. 
    """
    output = of.helper.general.add_file_appendix(dfile, "geo")
    with open(output, 'w') as geo_file:
        # Define the rectangle domain
        geo_file.write('lc_length = {:.4f}; // Element size for length\n'.format(esize_length))
        geo_file.write('lc_width = {:.4f}; // Element size for width\n'.format(esize_width))
        geo_file.write('length = {:.4f};\n'.format(length))
        geo_file.write('width = {:.4f};\n'.format(width))
        geo_file.write('Point(1) = {0, 0, 0, lc_width};\n')
        geo_file.write('Point(2) = {length, 0, 0, lc_length};\n')
        geo_file.write('Point(3) = {length, width, 0, lc_width};\n')
        geo_file.write('Point(4) = {0, width, 0, lc_length};\n')
        geo_file.write('Line(1) = {1, 2};\n')
        geo_file.write('Line(2) = {2, 3};\n')
        geo_file.write('Line(3) = {3, 4};\n')
        geo_file.write('Line(4) = {4, 1};\n')
        geo_file.write('Line Loop(5) = {1, 2, 3, 4};\n')
        geo_file.write('Plane Surface(6) = {5};\n')
        # Assign physical names to the facets
        geo_file.write('Physical Line(1) = {1}; // Left side\n')
        geo_file.write('Physical Line(2) = {2}; // Bottom side\n')
        geo_file.write('Physical Line(3) = {3}; // Right side\n')
        geo_file.write('Physical Line(4) = {4}; // Top side\n')
        geo_file.write('Physical Surface(5) = {6}; // Rectangle domain\n')
    done = of.helper.general.run_shell_command("gmsh -2 " + output)
    of.helper.io.msh2xdmf(dfile, dfile + "/", correct_gmsh=True)
    _, facet_function = of.helper.io.getXDMF(dfile + "/")
    g = of.simulation.problem.Geometry()
    g.mesh = facet_function.mesh()
    g.facets = facet_function
    g.dim = g.mesh.geometric_dimension()
    return g


study = of.Study("01_adaptive_growth")
p = of.simulation.Problem()
p.param.gen.title = "2D_Rectangle"
der_file = study.der_dir + p.param.gen.title
p.geom = create_Rectangle(100, 100, 5.0, 5.0, der_file)  # mm
#
# general info
p.param.gen.flag_defSplit = False
p.param.gen.output_file = of.helper.io.set_output_file(study.sol_dir + p.param.gen.title + "/TPM")
# time parameters
p.param.time.T_end = 3600 * 24.0 * 100.0  # 50 d
p.param.time.output_interval = 3600 * 3.0  # 0.5 d
p.param.time.dt = 3600 * 3.0  # 3 h
# material parameters base model
p.param.mat.rhoSR = 1190.0 * 1e-9  # kg / mm^3
p.param.mat.rhoFR = 993.3 * 1e-9  # kg / mm^3
p.param.mat.gammaFR = 1.0  # leave on 1.0!
p.param.mat.R = 8.31446261815324 * 1000  # (N mm) / (mol K)
p.param.mat.Theta = 37.0 + 273.15  # K
p.param.mat.lambdaS = 0.03312  # N / mm^2
p.param.mat.muS = 0.00662  # N / mm^2
p.param.mat.kF = 5.0e-11  # mm / s
p.param.mat.healthy_brain_nS = 0.75
# FEM parameters
p.param.fem.solver_type = "mumps"
p.param.fem.maxIter = 20
p.param.fem.rel = 1E-11
p.param.fem.abs = 1E-12
# ADDITIVES material parameter
molFt = 1.3e13  # kg / mol
DFt = 5.0e-3  # mm^2/s

p.param.add.prim_vars = []  # ["cFt"]
p.param.add.ele_types = []  # ["CG"]
p.param.add.ele_orders = []  # [1] 
p.param.add.tensor_orders = []  # [0]
p.param.add.molFkappa = []  # [molFt]
p.param.add.DFkappa = []  # [DFt]

model = of.simulation.base_models.TwoPhaseModel()
model.set_param(p)
model.set_function_spaces()
# initial conditions
p.param.init.uS_0S = [0.0, 0.0]
p.param.init.p_0S = 0.0
p.param.init.nS_0S = 0.75
field = df.Expression("c0*exp(-a*(pow((x[0]-x_s),2)+pow((x[1]-y_s),2)))", 
                      degree=2, c0=1.0e-1, a=0.0008, x_s=0.0, y_s=0.0)  # mmol / l
cFt_0S = df.interpolate(field, model.CG1_sca)
p.param.add.cFkappa_0S = []  # [cFt_0S]
# Bio chemical set up
bio_model = of.simulation.process_models.Simple()
bio_model.set_input(model.ansatz_functions)
bio_model.speed_nS = df.Expression("c0*exp(-a*(pow((x[0]-x_s),2)+pow((x[1]-y_s),2)))", 
                      degree=2, c0=-1.0e-7, a=0.0008, x_s=0.0, y_s=0.0) 
prod_list = bio_model.get_output()
model.set_process_models(prod_list)
# Boundary conditions
bc = []
fs = model.function_space
bc.append(df.DirichletBC(fs.sub(0).sub(1), 0.0, p.geom.facets, 1))  # 1: unstruc
bc.append(df.DirichletBC(fs.sub(0).sub(0), 0.0, p.geom.facets, 4))  # 2: unstruc
bc.append(df.DirichletBC(fs.sub(1), 0.0, p.geom.facets, 2))
bc.append(df.DirichletBC(fs.sub(1), 0.0, p.geom.facets, 3))
model.set_boundaries(bc, None)
# Set up model and begin to solve
model.set_structural_parameters()
model.set_weak_form()
df.set_log_level(30)
model.set_solver()
model.set_initial_conditions(p.param.init, p.param.add)
model.solve()
