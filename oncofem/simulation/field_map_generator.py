"""
Definition of the field map generator. Herein, the gathered information collected from pre-processing is translated to
the base model. Therefore, initial conditions and heterogeneities can be set and the problem can be defined.

Classes:
    Field_map_generator:    The field map generator interprets the given input data and creates mathematical objects 
                            with respect to the chosen model.
"""
import oncofem as of
import numpy as np
import fsl
import nibabel as nib
from scipy.interpolate import griddata
from skimage.segmentation import find_boundaries
from skimage.measure import regionprops

class FieldMapGenerator:
    """
    The field map generator interprets the given input data and creates mathematical objects with respect to the 
    chosen model.

    *Methods*:
        generate_geometry_file: Generates the geometry file
        mark_facet: Marks the facets made by bounding boxes
        interpolate_segm: interpolates segmentation of image file and creates an image
        map_field: Maps a field from image file onto geometry file
        run_edema_mapping: Runs edema mapping, interpolates edema segmentation and maps field onto geometry
        run_solid_tumor_mapping: Runs solid tumor entities (necrotic and active part), interpolation and mapping
        set_mixed_masks: Sets the handling of white matter mapping of tumor area.
        run_wm_mapping: Maps the fields of white and grey matter and cerebrospinal fluid
    """
    def __init__(self, mri=None):
        self.work_dir = None
        self.fmap_dir = None
        self.prim_mri_mod = None
        self.mixed_wm_mask = None
        self.mixed_gm_mask = None
        self.mixed_csf_mask = None
        self.xdmf_file = None
        self.surf_xdmf_file = None
        self.dolfin_mesh = None
        self.mapped_ede_file = None
        self.mapped_act_file = None
        self.mapped_nec_file = None
        self.mapped_wm_file = None
        self.mapped_gm_file = None
        self.mapped_csf_file = None
        self.interpolation_method = "linear"
        self.structure_mapping_method = "const_wm"
        self.volume_resolution = 16
        self.edema_max_value = 2.0
        self.edema_min_value = 1.0
        self.active_max_value = 2.0
        self.active_min_value = 1.0
        self.necrotic_max_value = 2.0
        self.necrotic_min_value = 1.0
        if mri is None:
            self.mri = None
        else:
            self.set_mri(mri)

    def set_mri(self, mri):
        """
        Sets state with working directory, loads measures, checks full modality and sets the affine.
        """
        if type(mri) is not str:
            self.mri = mri
            self.work_dir = mri.work_dir
            self.fmap_dir = self.work_dir + of.FIELD_MAP_PATH
        else:
            self.fmap_dir = mri
        of.helper.general.mkdir_if_not_exist(self.fmap_dir)
        self.stl_file = self.fmap_dir + "geometry.stl"
        self.mesh_file = self.fmap_dir + "geometry.mesh"

    def interpolate(self, image, name:str, plateau=None, hole=None, min_value:float=1.0, 
                    max_value:float=2.0, rest_value:float=0.0, method:str="linear") -> str: 
        """
        Interpolates a segmentation in between minimum and maximum value. Can also handle plateaus and holes. 

        *Arguments*:
            image: Input image in nifti format
            name: String for output file
            plateau: Image of plateau area, value will be set to maximum
            hole: Image of hole area, value will be set to zero
            min_value: float of minimal value at outer surface
            max_value: float of maximum value at center
            rest_value: float of surrounding tissue
            method: interpolation method, nearest or linear

        *Example*:
            output_file = interpolate_segm("edema.nii.gz", "edema")
        """
        if plateau is None:
            closed_vol = image
            center = regionprops(image.astype(int))[0].centroid
            coords_max = (np.array([int(center[0])]), np.array([int(center[1])]), np.array([int(center[2])]))
        else:
            closed_vol = image + plateau
            coords_max = np.where(plateau == 1)
        if hole is not None:
            max_bound = find_boundaries(hole.astype(int), mode="inner")
            coords_max = np.where(max_bound == 1)

        domain_shape = image.shape
        min_bound = find_boundaries(closed_vol.astype(int), mode="outer")
        coords_min = np.where(min_bound == 1)

        max_mask = np.zeros(domain_shape, dtype=bool)
        for i in range(len(coords_max[0])):
            max_mask[(coords_max[0][i], coords_max[0][i], coords_max[0][i])] = True

        # Generate coordinates for the occupied and unoccupied points
        coords_interp = np.where(~(min_bound | max_mask))
        coords_inverse = np.where(closed_vol == 0)
        if hole is not None:
            coords_inverse += np.where(hole == 1)

        # Create arrays of coordinates for the occupied points
        coords_occupied_min = np.array(coords_min).T
        coords_occupied_max = np.array(coords_max).T

        # Create an array of values for the occupied points
        values_occupied_min = np.full(coords_occupied_min.shape[0], min_value)
        values_occupied_max = np.full(coords_occupied_max.shape[0], max_value)
        mask = np.in1d(values_occupied_min, values_occupied_max).reshape(values_occupied_min.shape[0], -1).all(axis=1)
        values_occupied_min = values_occupied_min[~mask]

        # Perform the interpolation
        print("start interpolation")
        coords_occupied = np.concatenate((coords_occupied_min, coords_occupied_max))
        values_occupied = np.concatenate((values_occupied_min, values_occupied_max))
        interp_values = griddata(coords_occupied, values_occupied, coords_interp, method=method, fill_value=0.0)
        print("finished interpolation")
        # Create a 3D array with the minimum value
        values = np.full(domain_shape, 0.0)
        # Update the values array with the interpolated values
        coords_outside = (coords_inverse[0], coords_inverse[1], coords_inverse[2])
        values[coords_interp] = interp_values
        values[coords_max] = max_value
        values[coords_outside] = rest_value

        file_output = self.fmap_dir + name
        of.helper.io.write_field2nii(values, file_output, self.mri.affine)
        return file_output + ".nii.gz"

    def generate_geometry_file(self, primary_mri_mod: str) -> None:
        """
        Generates the geometry file of a given MRI modality. 

        *Arguments*:
            primary_mri_mod: String of input file in nifti format

        *Example*:
            generate_geometry_file("t1.nii.gz")
        """
        self.prim_mri_mod = primary_mri_mod  
        # first nii2stl
        of.helper.io.nii2stl(self.prim_mri_mod, self.stl_file, 0, self.fmap_dir)
        # second stl2mesh
        of.helper.io.stl2mesh(self.stl_file, self.mesh_file, self.volume_resolution)
        # third msh2xmdf
        self.xdmf_file = of.helper.io.mesh2xdmf(self.mesh_file, self.fmap_dir)
        # load mesh
        self.dolfin_mesh = of.helper.io.load_mesh(self.xdmf_file)

    def run_edema_mapping(self) -> None:
        """
        Interpolates edema and maps onto geometry 

        *Example*:
            run_edema_mapping()
        """
        plateau = self.mri.act_mask + self.mri.nec_mask
        ede_ip = self.interpolate(self.mri.ede_mask, "edema_ip", plateau=plateau,
                                  min_value=self.edema_min_value, max_value=self.edema_max_value,
                                  method=self.interpolation_method)
        self.mapped_ede_file = of.helper.io.map_field(ede_ip, self.fmap_dir + "edema", self.dolfin_mesh)

    def run_solid_tumor_mapping(self) -> None:
        """
        Interpolates solid tumor entities (necrotic and active part) and maps onto geometry 

        *Example*:
            run_solid_tumor_mapping()
        """
        # Needed to change edema with necrotic...somehow lead to overwriting of edema
        # generate separated nii maps
        act_ip = self.interpolate(self.mri.act_mask, "active_ip", hole=self.mri.nec_mask, 
                                  min_value=self.active_min_value, max_value=self.active_max_value, 
                                  method=self.interpolation_method)

        # hotfix, necrotic image has not nicely convex hull
        max_bound = find_boundaries((self.mri.nec_mask + self.mri.act_mask).astype(int), mode="inner")
        solid_tumor = nib.Nifti1Image(self.mri.nec_mask + self.mri.act_mask - max_bound, self.mri.affine)
        active_tumor = nib.load(act_ip)
        nec = fsl.wrappers.fslmaths(active_tumor).div(active_tumor).mul(-1).add(solid_tumor).run()
        nib.save(nec, self.fmap_dir + "necrotic_ip.nii.gz")

        self.mapped_act_file = of.helper.io.map_field(act_ip, self.fmap_dir + "active", self.dolfin_mesh)
        self.mapped_nec_file = of.helper.io.map_field(self.fmap_dir + "necrotic_ip.nii.gz", 
                                                      self.fmap_dir + "necrotic", self.dolfin_mesh)

    def set_mixed_masks(self, classes:list[np.ndarray]=None) -> None:
        """
        Sets tumor classes analogous to the white and gray matter and csf. Needed for mean averaged value. List
        should have three entities. First for white matter, second for gray matter, third for csf.

        *Arguments*:
            classes: List of tumor class images, optional for "mean_averaged_value" white matter mapping method

        *Example*:
            set_mixed_masks()        
        """
        if self.structure_mapping_method == "const_wm":
            tumor_mask = nib.Nifti1Image(self.mri.act_mask + self.mri.nec_mask + self.mri.ede_mask, self.mri.affine)
            fsl.wrappers.fslmaths(self.mri.wm_mask).add(tumor_mask).run(self.fmap_dir + "wm.nii.gz")
            self.mixed_wm_mask = self.fmap_dir + "wm.nii.gz"
            self.mixed_gm_mask = self.mri.gm_mask
            self.mixed_csf_mask = self.mri.csf_mask

        elif self.structure_mapping_method == "mean_averaged_value":
            fsl.wrappers.fslmaths(self.mri.wm_mask).add(classes[0]).run(self.fmap_dir + "wm.nii.gz")
            fsl.wrappers.fslmaths(self.mri.gm_mask).add(classes[1]).run(self.fmap_dir + "gm.nii.gz")
            fsl.wrappers.fslmaths(self.mri.csf_mask).add(classes[2]).run(self.fmap_dir + "csf.nii.gz")
            self.mixed_wm_mask = self.fmap_dir + "wm.nii.gz"
            self.mixed_gm_mask = self.fmap_dir + "gm.nii.gz"
            self.mixed_csf_mask = self.fmap_dir + "csf.nii.gz"

        elif self.structure_mapping_method == "tumor_entity_weighted":
            print("not implemented")
            pass

    def run_structure_mapping(self) -> None:
        """
        Maps white matter fields (white and grey and csf) onto geometry 

        *Example*:
            run_wm_mapping()
        """
        self.mapped_wm_file = of.helper.io.map_field(self.mixed_wm_mask, self.fmap_dir + "white_matter", self.xdmf_file)
        self.mapped_gm_file = of.helper.io.map_field(self.mixed_gm_mask, self.fmap_dir + "gray_matter", self.xdmf_file)
        self.mapped_csf_file = of.helper.io.map_field(self.mixed_csf_mask, self.fmap_dir + "csf", self.xdmf_file)
