"""
Nii2Mesh interface
"""

from oncofem.helper import constant as const
from oncofem.helper import general as gen
import os, subprocess

class Nii2Mesh:


    """
    Options
    -a s    atlas text file (e.g. '-a D99_v2.0_labels_semicolon.txt')
    -b v    bubble fill (0=bubbles included, 1=bubbles filled, default 0)
    -i v    isosurface intensity (d=dark, m=mid, b=bright, number for custom, default medium)
    -l v    only keep largest cluster (0=all, 1=largest, default 1)
    -o v    Original marching cubes (0=Improved Lewiner, 1=Original, default 0)
    -p v    pre-smoothing (0=skip, 1=smooth, default 1)
    -r v    reduction factor (default 0.25)
    -q v    quality (0=fast, 1= balanced, 2=best, default 1)
    -s v    post-smoothing iterations (default 0)
    -v v    verbose (0=silent, 1=verbose, default 0)
    mesh extension sets format (.gii, .json, .mz3, .obj, .ply, .pial, .stl, .vtk)
    Example: './src/nii2mesh voxels.nii mesh.obj'
    Example: './src/nii2mesh bet.nii.gz -i 22 myOutput.obj'
    Example: './src/nii2mesh bet.nii.gz -i b bright.obj'
    Example: './src/nii2mesh img.nii -v 1 out.ply'
    Example: './src/nii2mesh img.nii -p 0 -r 1 large.ply'
    Example: './src/nii2mesh img.nii -r 0.1 small.gii'
    """
    def __init__(self):
        self.atlas = None
        self.bubble_fill = 0
        self.iso_surface = "m"
        self.largest_cluster = 1
        self.original_marching_cubes = 0
        self.pre_smoothing = 1
        self.reduction_factor = 0.25
        self.quality = 1
        self.post_smoothing = 0
        self.verbose = 0
        self.input = None
        self.output = None

    def run_nii2mesh(self):
        """
        converts nii file to mesh, further info in class.
        """
        bashCmd = [const.NII2MESH_DIR+"./nii2mesh"]
        bashCmd.append(self.input)
        if self.atlas is not None:
            bashCmd.append("-a")
            bashCmd.append(self.atlas)
        bashCmd.append("-b")
        bashCmd.append(str(self.bubble_fill))
        bashCmd.append("-i")
        bashCmd.append(str(self.iso_surface))
        bashCmd.append("-l")
        bashCmd.append(str(self.largest_cluster))
        bashCmd.append("-o")
        bashCmd.append(str(self.original_marching_cubes))
        bashCmd.append("-p")
        bashCmd.append(str(self.pre_smoothing))
        bashCmd.append("-r")
        bashCmd.append(str(self.reduction_factor))
        bashCmd.append("-q")
        bashCmd.append(str(self.quality))
        bashCmd.append("-s")
        bashCmd.append(str(self.post_smoothing))
        bashCmd.append("-v")
        bashCmd.append(str(self.verbose))
        bashCmd.append(self.output)
        head, tail = os.path.split(self.output)
        gen.mkdir_if_not_exist(head)
        subprocess.Popen(bashCmd, stdout=subprocess.PIPE)
