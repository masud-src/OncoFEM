"""
# **************************************************************************#
#                                                                           #
# === Greedy module  =======================================================#
#                                                                           #
# **************************************************************************#
# In this module an interface to the greedy package is implemented.
# With this the user can perform co-registrations with medical images
# 
#
# Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
#
# --------------------------------------------------------------------------#
"""

from oncofem.helper.general import splitPath
from oncofem.helper import constant
from os import sep
from pathlib import Path
import subprocess

class Greedy(object):
    """ 
    greedy: Paul's greedy diffeomorphic registration implementation
    Usage: 
        greedy [options]
    Required options: 
        -d DIM                 : Number of image dimensions
        -i fix.nii mov.nii     : Image pair (may be repeated)
        -o <file>              : Output file (matrix in affine mode; image in deformable mode, 
                                 metric computation mode; ignored in reslicing mode)
    Mode specification: 
        -a                     : Perform affine registration and save to output (-o)
        -brute radius          : Perform a brute force search around each voxel 
        -moments <1|2>         : Perform moments of inertia rigid alignment of given order.
                                     order 1 matches center of mass only
                                     order 2 matches second-order moments of inertia tensors
        -r [tran_spec]         : Reslice images instead of doing registration 
                                     tran_spec is a series of warps, affine matrices
        -iw inwarp outwarp     : Invert previously computed warp
        -root inwarp outwarp N : Convert 2^N-th root of a warp 
        -jac inwarp outjac     : Compute the Jacobian determinant of the warp 
        -metric                : Compute metric between images
    Options in deformable / affine mode: 
        -w weight              : weight of the next -i pair
        -m metric              : metric for the entire registration
                                            SSD:          sum of square differences (default)
                                            MI:           mutual information
                                            NMI:          normalized mutual information
                                            NCC <radius>: normalized cross-correlation
                                            MAHAL:        Mahalanobis distance to target warp
        -e epsilon             : step size (default = 1.0), 
                                     may also be specified per level (e.g. 0.3x0.1)
        -n NxNxN               : number of iterations per level of multi-res (100x100) 
        -threads N             : set the number of allowed concurrent threads
        -gm mask.nii           : fixed image mask (metric gradients computed only over the mask)
        -gm-trim <radius>      : generate the fixed image mask by trimming the extent
                                 of the fixed image by given radius. This is useful during affine
                                 registration with the NCC metric when the background of your images
                                 is non-zero. The radius should match that of the NCC metric.
        -mm mask.nii           : moving image mask (pixels outside are excluded from metric computation)
        -ncc-mask-dilate       : flag, specifies that fixed and moving masks should be dilated by the radius
                                 of the NCC/WNCC metric during registration. This is for when your mask goes
                                 up to the edge of the tissue and you want the tissue/background edge to count
                                 towards the metric.
    Defining a reference space for registration (primarily in deformable mode): 
        -ref <image>           : Use supplied image, rather than fixed image to define the reference space
        -ref-pad <radius>      : Define the reference space by padding the fixed image by radius. Useful when
                                 the stuff you want to register is at the border of the fixed image.
        -bg <float|NaN>        : When mapping fixed and moving images to reference space, fill missing values
                                 with specified value (default: 0). Passing NaN creates a mask that excludes
                                 missing values from the registration. This value is also used when computing
                                 the metric at a pixel that maps outide of the moving image or moving mask
        -it filenames          : Specify transforms (matrices, warps) that map moving image to reference space.
                                 Typically used to supply an affine transform when running deformable registration.
                                 Different from -ia, which specifies the initial transform for affine registration.
    Specific to deformable mode: 
        -tscale MODE           : time step behavior mode: CONST, SCALE [def], SCALEDOWN
        -s sigma1 sigma2       : smoothing for the greedy update step. Must specify units,
                                 either `vox` or `mm`. Default: 1.732vox, 0.7071vox
        -oinv image.nii        : compute and write the inverse of the warp field into image.nii
        -oroot image.nii       : compute and write the (2^N-th) root of the warp field into image.nii, where
                                 N is the value of the -exp option. In stational velocity mode, it is advised
                                 to output the root warp, since it is used internally to represent the deformation
        -wp VALUE              : Saved warp precision (in voxels; def=0.1; 0 for no compression).
        -noise VALUE           : Standard deviation of white noise added to moving/fixed images when 
                                 using NCC metric. Relative to intensity range. Def=0.001
        -exp N                 : The exponent used for warp inversion, root computation, and in stationary 
                                 velocity field (Diff Demons) mode. N is a positive integer (default = 6) 
        -sv                    : Performs registration using the stationary velocity model, similar to diffeomoprhic 
                                 Demons (Vercauteren 2008 MICCAI). Internally, the deformation field is 
                                 represented as 2^N self-compositions of a small deformation and 
                                 greedy updates are applied to this deformation. N is specified with the -exp 
                                 option (6 is a good number). This mode results in better behaved
                                 deformation fields and Jacobians than the pure greedy approach.
        -svlb                  : Same as -sv but uses the more accurate but also more expensive 
                                 update of v, v <- v + u + [v,u]. Experimental feature 
        -sv-incompr            : Incompressibility mode, implements Mansi et al. 2011 iLogDemons
        -id image.nii          : Specifies the initial warp to start iteration from. In stationary mode, this 
                                 is the initial stationary velocity field (output by -oroot option)
    Initial transform specification (for affine mode): 
        -ia filename           : initial affine matrix for optimization (not the same as -it) 
        -ia-identity           : initialize affine matrix based on NIFTI headers 
        -ia-image-centers      : initialize affine matrix based on matching image centers 
        -ia-image-side CODE    : initialize affine matrix based on matching center of one image side 
        -ia-moments <1|2>      : initialize affine matrix based on matching moments of inertia
    Specific to affine mode (-a):
        -dof N                 : Degrees of freedom for affine reg. 6=rigid, 12=affine
        -jitter sigma          : Jitter (in voxel units) applied to sample points (def: 0.5)
        -search N <rot> <tran> : Random search over rigid transforms (N iter) before starting optimization
                                 'rot' may be the standard deviation of the random rotation angle (degrees) or 
                                 keyword 'any' (any rotation) or 'flip' (any rotation or flip). 
                                 'tran' is the standard deviation of the random offset, in physical units. 
    Specific to moments of inertia mode (-moments 2): 
        -det <-1|1>            : Force the determinant of transform to be either 1 (no flip) or -1 (flip)
        -cov-id                : Assume identity covariance (match centers and do flips only, no rotation)
    Specific to reslice mode (-r): 
        -rf fixed.nii          : fixed image for reslicing
        -rm mov.nii out.nii    : moving/output image pair (may be repeated)
        -rs mov.vtk out.vtk    : moving/output surface pair (vertices are warped from fixed space to moving)
        -ri interp_mode        : interpolation for the next pair (NN, LINEAR*, LABEL sigma)
        -rb value              : background (i.e. outside) intensity for the next pair (default 0)
        -rc outwarp            : write composed transforms to outwarp 
        -rj outjacobian        : write Jacobian determinant image to outjacobian 
    Specific to metric computation mode (-metric): 
        -og out.nii            : write the gradient of the metric to file
    For developers: 
        -debug-deriv           : enable periodic checks of derivatives (debug) 
        -debug-deriv-eps       : epsilon for derivative debugging 
        -debug-aff-obj         : plot affine objective in neighborhood of -ia matrix 
        -dump-pyramid          : dump the image pyramid at the start of the registration
        -dump-moving           : dump moving image at each iter
        -dump-freq N           : dump frequency
        -dump-prefix <string>  : prefix for dump files (may be a path) 
        -powell                : use Powell's method instead of LGBFS
        -float                 : use single precision floating point (off by default)
        -version               : print version info
        -V <level>             : set verbosity level (0: none, 1: default, 2: verbose)
    Environment variables: 
        GREEDY_DATA_ROOT       : if set, filenames can be specified relative to this path
    """

    def __init__(self, study):
        self.study_dir = study
        self.d = "3"
        self.i = None
        self.o = None
        self.a = None
        self.br = None
        self.moments = None
        self.r = None
        self.iw = None
        self.root = None
        self.jac = None
        self.metric = None
        self.w = None
        self.m = "NCC 4x4x4"
        self.e = None
        self.n = "100x50x10"
        self.threads = None
        self.gm = None
        self.gm_trim = None
        self.mm = None
        self.ncc_mask_dilate = None
        self.ref = None
        self.ref_pad = None
        self.bg = None
        self.it = None
        self.tscale = None
        self.s = None
        self.oinv = None
        self.oroot = None
        self.wp = None
        self.noise = None
        self.exp = None
        self.sv = None
        self.svlb = None
        self.sv_incompr = None
        self.id = None
        self.ia = None
        self.ia_identity = None
        self.ia_image_centers = None
        self.ia_image_side = None
        self.ia_moments = None
        self.dof = None
        self.jitter = None
        self.search = None
        self.det = None
        self.cov_id = None
        self.rf = None
        self.rm = None
        self.rs = None
        self.ri = "LINEAR"
        self.rb = None
        self.rc = None
        self.rj = None
        self.og = None
        self.debug_deriv = None
        self.debug_deriv_eps = None
        self.debug_aff_obj = None
        self.dump_pyramid = None
        self.dump_moving = None
        self.dump_freq = None
        self.dump_prefix = None
        self.powell = None
        self.float = None
        self.version = None
        self.V = None

        self.root_command = constant.GREEDY_DIR + "greedy"

    def coregister_patient2sri24(self, input_file):  
        """
        Co-register patient data to SRI24 Atlas. Works with two steps. First rigid transformation matrix is set up. Then this transformation is applied.
        """

        file, path = splitPath(input_file)
        file_wo_extension = Path(Path(input_file).stem).stem
        file_rigid_trafo = "rigid_" + str(file_wo_extension) + ".mat"
        file_bc = file_wo_extension + "_bc.nii.gz"
        file_sri24 = file_wo_extension + "_sri24.nii.gz"

        # Step 1: Generate Trafo File
        command = [self.root_command]
        command.append("-d")
        command.append(self.d)
        command.append("-a")
        command.append("-dof")
        command.append("6")
        command.append("-m")
        command.append(self.m)
        command.append("-i")
        command.append(path + sep + file_bc)
        command.append(constant.PATH_SRI24_T1)
        command.append("-o")
        command.append(path + sep + file_rigid_trafo)
        command.append("-ia-image-centers")
        command.append("-n")
        command.append(self.n)
        p = subprocess.Popen(command, stdout=subprocess.PIPE)
        print(p.communicate())

        # Step 2: Apply Trafo file
        command = [self.root_command]
        command.append("-d")
        command.append(self.d)
        command.append("-rf")
        command.append(constant.PATH_SRI24_T1)
        command.append("-ri")
        command.append(self.ri)
        command.append("-rm")
        command.append(input_file)
        command.append(path + sep + file_sri24)
        command.append("-r")
        command.append(path + sep + file_rigid_trafo)
        p = subprocess.Popen(command, stdout=subprocess.PIPE)
        print(p.communicate())

    def test_co_registersri24(self, input_file, modality):
        file, path = splitPath(input_file)
        file_wo_extension = Path(Path(input_file).stem).stem
        file_sri24 = file_wo_extension + "_sri24.nii.gz"
        command = ["/home/marlon/Software/CaPTk/1.8.1/captk"]
        command.append("Preprocessing.cwl")
        command.append("-i")
        command.append(input_file)
        command.append("-rFI")
        if modality == "T2": 
            command.append(constant.PATH_SRI24_T2) 
        else: 
            command.append(constant.PATH_SRI24_T1) 
        command.append("-o")
        command.append(path + sep + file_sri24)
        command.append("-reg")
        command.append("RIGID")
        p = subprocess.Popen(command, stdout=subprocess.PIPE)
        print(p.communicate())

    def deformable_registration(self, input_file):
        path = splitPath(input_file)[1]
        command = [self.root_command]
        command.append("-m")
        command.append(self.m)
        command.append("-i")
        command.append(input_file)
        command.append(constant.PATH_SRI24_T1)
        command.append("-it")
        command.append(path + sep + "rigid.mat")
        command.append("-o")
        command.append(path + sep + "warp.nii.gz")
        command.append("-oinv")
        command.append(path + sep + "inverse_warp.nii.gz")
        command.append("-n")
        command.append(self.n)
        p = subprocess.Popen(command, stdout=subprocess.PIPE)
        print(p.communicate())

    def apply_warp_patient2sri24(self, input_file):
        path = splitPath(input_file)[1]
        command = [self.root_command]
        command.append("-rf")
        command.append(constant.PATH_SRI24_T1)
        command.append("-rm")
        command.append(input_file)
        command.append(input_file.replace('.nii', '_sri24.nii'))
        command.append("-r")
        command.append(path + sep + "warp.nii.gz")
        command.append(path+sep+"rigid.mat")
        p = subprocess.Popen(command, stdout=subprocess.PIPE)
        print(p.communicate())
