"""
# **************************************************************************#
#                                                                           #
# === field map generator package  =========================================#
#                                                                           #
# **************************************************************************#
# In this sub-package of oncofem the field map generator is implemented.
# Herein, the following sub-modules can be found:
# 
#   field_map_generator - main module
#           
#       This module can be used as controller module for sub-declared tasks.
#       The main functionalies are: 
#           - the generation of xdmf files from nifti inputs,
#           - marking areas of the surface for boundary conditions,
#           - map the tumor compartments onto the generated mesh,
#           - map the heterogeneous distribution of white and gray matter
#             and csf onto the generated mesh, and
#           - map arbitrary fields onto the generated mesh.
# 
#   geometry
#       
#       The geometry module holds the Geometry class. Herein, the geometry
#       of a problem can be defined and elementary information about it is
#       gathered and can be collected in that entity. Furthermore, simple
#       academic examples can be created with predefined functions.
#
#
# Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
#
# --------------------------------------------------------------------------#
"""
