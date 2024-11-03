# OncoFEM

import os
import os.path
import pathlib
import configparser
import dolfin as df
import ufl
import numpy as np
from abc import ABC
from typing import Union, Generator, Any
import subprocess
import shlex
from pathlib import Path
import gzip
import shutil
import meshio
import pandas as pd
import matplotlib.pyplot as plt
import ast
import nibabel as nib
import SVMTK as svmtk
import nibabel.loadsave
from skimage import measure
from scipy.ndimage import gaussian_filter
from stl import mesh
import time
from scipy.interpolate import griddata
from skimage.segmentation import find_boundaries
from skimage.measure import regionprops
import copy

pip install nibabel numpy scikit-image numpy-stl