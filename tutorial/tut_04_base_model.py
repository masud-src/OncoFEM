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
import os
import oncofem.struc as str
from oncofem.struc.problem import Problem
from tutorial.data import academic_geometries

# define study
study = str.Study("paper_model")
x = Problem()

# geometry
x.param.gen.title = "2D_CircleRectangle"
x.geom.dim = 2
der_file = study.der_dir + x.param.gen.title
der_path = der_file + os.sep
x.geom.mesh, x.geom.facet_function, area_conc, area_df = academic_geometries.create_2D_QuarterCircle_Tumor(0.0001, 1000.0, 1.0, 0.0006, 40, der_file, der_path, 1.15E-13, 1e-5)  # 0.01 60
