"""
Beispiel Fieldmapping 2D (poisson in tut file)
Beispiel Fieldmapping 3D (poisson in tut file)

In this file the functionalities of the field map generator are tested. In
order to perform the tests either the default data of the BraTS 2020 data
set can be used, or the user can change the input data for custom testing. 
Following tests are implemented:

  - Generation of an academic test geometry,
  - Generation and modifying of a mesh from input nifti data,
  - Marking of subdomain surface for boundary conditions,
  - Creation of interpolated distributions of tumour segmentation and 
    brain heterogenities,
  - Mapping of a tumor segmentation and brain heterogenities onto the mesh. 


The generated test geometries are tested with the weak form of a simple 
heat equation. Therefore, first the test study folder is created in the
study sub-directory. Its name is "test_field_map_generator".
In a next step, several geometries and corresponding boundary conditions 
are created. These are:

  1. Quarter circle geometry with Dirichlet boundary condition at the 
     outer surface.
  2. Quarter ring geometry with Dirichlet boundary condition at inner
     surface and Neumann boundary condition at outer surface.
  3. Brain geometry with Dirichlet boundary conditions at brain stem.
  4. Brain geometry with Dirichlet boundary conditions of tumor 
     segmentation.
  5. Brain geometry with Dirichlet boundary conditions of tumor
     segmentation and heterogeneous distribution of permeability factor.

The brain geometries are first modified within their native nifti format.
In a second step the distributions are mapped onto the particular geometry.
All of the prepared geometries are then used as input for the evaluation
the transient heat equation. Therefore, the created mesh and boundary
conditions are handed over to a function that generates a fenics test case
for the weak form of 

Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""
# Imports
import dolfin as df
import ufl
import datetime
import os
import oncofem as of

study = of.Study("tut_03")
subj_1 = study.create_subject("Subject_1")
state_1 = subj_1.create_state("init_state")
measure_1 = state_1.create_measure("data/BraTS/BraTS20_Training_001/BraTS20_Training_001_t1.nii.gz", "t1")
measure_2 = state_1.create_measure("data/BraTS/BraTS20_Training_001/BraTS20_Training_001_t1ce.nii.gz", "t1ce")
measure_3 = state_1.create_measure("data/BraTS/BraTS20_Training_001/BraTS20_Training_001_t2.nii.gz", "t2")
measure_4 = state_1.create_measure("data/BraTS/BraTS20_Training_001/BraTS20_Training_001_flair.nii.gz", "flair")
measure_5 = state_1.create_measure("data/BraTS/BraTS20_Training_001/BraTS20_Training_001_seg.nii.gz", "seg")
"""

"""
mri = of.MRI(state=state_1)
mri.load_measures()
mri.set_affine()
mri.tumor_segmentation.infer_param.output_path = "/home/marlon/Software/OncoFEM/tutorial/data/BraTS/BraTS20_Training_001/BraTS20_Training_001_seg.nii.gz"
mri.tumor_segmentation.set_compartment_masks()
"""
copy images, because already generalised
"""
run_cp = False
if run_cp:
    print("cp " + measure_1.dir_act + " " + mri.generalisation.generalisation_path + ".")
    of.helper.run_shell_command("cp " + measure_1.dir_act + " " + mri.generalisation.generalisation_path + ".")
"""
asdf
"""
p = of.Problem(mri)
"""
asdf
"""
# Field mapping
run_fmp = True
if run_fmp:
    fmap = of.modelling.FieldMapGenerator(p)
    fmap.set_fmap_dir(study.der_dir + subj_1.ident + os.sep + state_1.dir + "fmap" + os.sep)
    fmap.mri = mri
    # Set up geometry
    fmap.set_primary_mri_mod(mri.t1_dir)
    fmap.geom.volume_resolution = 2#20
    fmap.generate_geometry_file()
    fmap.load_mesh()
    b1 = of.modelling.field_map_generator.BoundingBox(fmap.geom.dolfin_mesh, (100.0, 129.0), (115.0, 160.0), (-20.0, 10.0))
    b2 = of.modelling.field_map_generator.BoundingBox(fmap.geom.dolfin_mesh, (78.0, 95.0), (154.0, 165.0), (-20.0, 20.0))
    p.geom.domain, p.geom.facet_function = fmap.mark_facet([b1, b2])
    p.geom.mesh = fmap.geom.dolfin_mesh
    p.geom.dim = 3
    # Set up tumour mapping
    #tmg = fmg.set_up_tumor_map_generator()
    #tmg.max_edema_value = 1.0E-4  # max concentration
    #tmg.max_solid_tumor_value = 0.4  # max solid tumor
    #tmg.max_necrotic_value = 0.5  # max necrotic core
    fmap.generate_tumor_map()
    p.geom.edema_distr = fmg.read_mapped_xdmf(fmg.mapped_edema_file)
    p.geom.solid_tumor_distr = fmg.read_mapped_xdmf(fmg.mapped_solid_tumor_file)
    p.geom.necrotic_distr = fmg.read_mapped_xdmf(fmg.mapped_necrotic_file)


#def function_space(mesh):
#    # Define function space
#    V = df.FunctionSpace(mesh, 'P', 1)
#    return V
#
## Define boundary condition
#def boundary(x, on_boundary):
#    return on_boundary
#bc = df.DirichletBC(V, df.Constant(0), boundary)
#
## Define initial value
#u_0 = df.Expression('exp(-a*pow(x[0], 2) - a*pow(x[1], 2))', degree=2, a=5)
#u_n = df.interpolate(u_0, V)
#
#def solve_heat_equation(V,  bc, ic, output_file):
#    T = 2.0            # final time
#    num_steps = 50     # number of time steps
#    dt = T / num_steps # time step size
#
#    u_n = df.interpolate(ic, V)
#
#    # Define variational problem
#    u = ufl.TrialFunction(V)
#    v = ufl.TestFunction(V)
#
#    F = u*v*df.dx + dt*ufl.dot(ufl.grad(u), ufl.grad(v))*df.dx - (u_n)*v*df.dx
#    a, L = ufl.lhs(F), ufl.rhs(F)
#
#    # Create VTK file for saving solution
#    xdmf_file = df.XDMFFile(output_file)
#
#    # Time-stepping
#    u = df.Function(V)
#    t = 0
#    for n in range(num_steps):
#        
#        # Update current time
#        t += dt
#        
#        # Compute solution
#        df.solve(a == L, u, bc)
#        
#        # Save to file and plot solution
#        u.rename("u", "u")
#        xdmf_file.write(u, t)
#
#        # Update previous solution
#        u_n.assign(u)
#