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

##############################################################################
# Generation of test study
study = of.Study("test_field_map_generator")
##############################################################################


subj = study.create_subject("UPENN-GBM-00002")
state = subj.create_state("state_1", datetime.date.today())

folder = "/media/marlon/data/MRI_data/UPENN-GBM/images_structural/UPENN-GBM-00002_11/"
#folder = "/media/marlon/data/MRI_data/UPENN-GBM/images_segm/"
state.create_measure(folder + "UPENN-GBM-00002_11_T1.nii.gz", "t1")
state.create_measure(folder + "UPENN-GBM-00002_11_T1GD.nii.gz", "t1ce")
state.create_measure(folder + "UPENN-GBM-00002_11_T2.nii.gz", "t2")
state.create_measure(folder + "UPENN-GBM-00002_11_FLAIR.nii.gz", "flair")
folder = "/media/marlon/data/MRI_data/UPENN-GBM/images_segm/"
state.create_measure(folder + "UPENN-GBM-00002_11_segm.nii.gz", "seg")
##############################################################################


def function_space(mesh):
    # Define function space
    V = df.FunctionSpace(mesh, 'P', 1)
    return V

# Define boundary condition
def boundary(x, on_boundary):
    return on_boundary
bc = df.DirichletBC(V, df.Constant(0), boundary)

# Define initial value
u_0 = df.Expression('exp(-a*pow(x[0], 2) - a*pow(x[1], 2))', degree=2, a=5)
u_n = df.interpolate(u_0, V)

def solve_heat_equation(V,  bc, ic, output_file):
    T = 2.0            # final time
    num_steps = 50     # number of time steps
    dt = T / num_steps # time step size

    u_n = df.interpolate(ic, V)

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    F = u*v*df.dx + dt*ufl.dot(ufl.grad(u), ufl.grad(v))*df.dx - (u_n)*v*df.dx
    a, L = ufl.lhs(F), ufl.rhs(F)

    # Create VTK file for saving solution
    xdmf_file = df.XDMFFile(output_file)

    # Time-stepping
    u = df.Function(V)
    t = 0
    for n in range(num_steps):
        
        # Update current time
        t += dt
        
        # Compute solution
        df.solve(a == L, u, bc)
        
        # Save to file and plot solution
        u.rename("u", "u")
        xdmf_file.write(u, t)

        # Update previous solution
        u_n.assign(u)
