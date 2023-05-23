# Imports
import os
from oncofem.struc.study import Study
from oncofem.modelling.field_map_generator import FieldMapGenerator
import dolfin as df
import academic_geometries

#Define Study
study = Study("resection")

folder = "/media/marlon/data/studies/paper_model/sol/2D_CircleRectangle/"
file = "TPM.xdmf"


title = "2D_CircleRectangle"
der_file = study.der_dir + title
mesh, _, _, _ = academic_geometries.create_2D_QuarterCircle_Tumor(0.0001, 1000.0, 1.0, 0.0006, 40, der_file, 1.15E-13, 1e-5)
V = df.FunctionSpace(mesh, 'P', 1)
f1 = df.Function(V)
f_in = df.XDMFFile(folder+file)

f_in.read_checkpoint(f1, "f", 0)
f_in.read_checkpoint(f1, "f", 1)

# Field mapping
subject_dir = study.der_dir + "W1" + os.sep
fmg = FieldMapGenerator(study)
# Set up geometry
fmg.set_general(t1_dir=t1_dir, work_dir=subject_dir)
fmg.volume_resolution = 8#20
fmg.generate_geometry_file()
domain, facet_function = fmg.mark_facet(x_bounds=(106.0, 129.0), y_bounds=(130, 148), z_bounds=(-2, 6))
# Set up tumour mapping
fmg.tumor_seg_file = tumor_seg_dir
tmg = fmg.set_up_tumor_map_generator()
tmg.max_edema_value = 1.0E-4  # max concentration
tmg.max_solid_tumor_value = 0.4  # max solid tumor
tmg.max_necrotic_value = 0.5  # max necrotic core
fmg.generate_tumor_map()
edema_distr = fmg.read_mapped_xdmf(fmg.mapped_edema_file)
solid_tumor_distr = fmg.read_mapped_xdmf(fmg.mapped_solid_tumor_file)
necrotic_distr = fmg.read_mapped_xdmf(fmg.mapped_necrotic_file)

x.geom.mesh = x.geom.facet_function.mesh()
x.geom.dx = df.Measure("dx", metadata={'quadrature_degree': 2})

xdmf = df.XDMFFile(x.geom.mesh.mpi_comm(), outfile + ".xdmf")
xdmf.parameters["flush_output"] = True
xdmf.parameters["functions_share_mesh"] = True
xdmf.write(mesh)
xdmf.write(regions)
xdmf.close()


