"""
Definition of the field map generator. Herein, the gathered information collected from pre-processing is translated to
the base model. Therefore, initial conditions and heterogeneities can be set and the problem can be defined.

Classes:
    Field_map_generator:    The field map generator interprets the given input data and creates mathematical objects 
                            with respect to the chosen model.
"""
from typing import Any, Union

import os
import copy
import nibabel as nib
import numpy as np
from scipy.interpolate import griddata
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries
import dolfin as df

import oncofem.utils.io as io

FIELD_MAP_PATH = "field_mapping/"


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

    def __init__(self, work_dir=None):
        self.work_dir = work_dir
        if work_dir is None:
            self.work_dir = os.getcwd() + os.sep
        if not work_dir.endswith(os.sep):
            self.work_dir = work_dir + os.sep
        self.out_dir = None
        self.volume_resolution = 16
        self.xdmf_file = None
        self.dolfin_mesh = None
        self.affine = None
        self.shape = None
        self.interpolation_method = "linear"
        self.tumor_class_mapping = {"edema": 2, "active": 4, "necrotic": 1}

        self.mixed_wm_mask = None
        self.mixed_gm_mask = None
        self.mixed_csf_mask = None
        self.mapped_ede_file = None
        self.mapped_act_file = None
        self.mapped_nec_file = None
        self.mapped_wm_file = None
        self.mapped_gm_file = None
        self.mapped_csf_file = None
        self.structure_mapping_method = "const_wm"
        self.edema_max_value = 2.0
        self.edema_min_value = 1.0
        self.active_max_value = 2.0
        self.active_min_value = 1.0
        self.necrotic_max_value = 2.0
        self.necrotic_min_value = 1.0

    def nii2dolfin_mesh(self, nii_input: str) -> df.Mesh:
        """
        Generates the geometry file of a given MRI modality.

        *Arguments*:
            primary_mri_mod: String of input file in nifti format

        *Example*:
            generate_geometry_file("t1.nii.gz")
        """
        self.out_dir = io.set_out_dir(self.work_dir, FIELD_MAP_PATH)
        _, _, name = io.get_path_file_extension(nii_input)
        stl_file = self.out_dir + name + ".stl"
        mesh_file = self.out_dir + name + ".mesh"
        # first nii2stl
        io.nii2stl(nii_input, stl_file, self.out_dir)
        # second stl2mesh
        io.stl2mesh(stl_file, mesh_file, self.volume_resolution)
        # third msh2xmdf
        self.xdmf_file = io.mesh2xdmf(mesh_file, self.out_dir)
        # load mesh
        self.dolfin_mesh = self.load_mesh(self.xdmf_file)
        return self.dolfin_mesh

    @staticmethod
    def load_mesh(file: str) -> df.Mesh:
        """
        Loads an XDMF file from file directory

        *Arguments*:
            file: String of XDMF file directory

        *Example*:
            xdmf_file = load_mesh("studies/test_study/der/geometry/geometry.xdmf")
        """
        mesh = df.Mesh()
        with df.XDMFFile(file) as infile:
            infile.read(mesh)
        return mesh

    @staticmethod
    def read_mapped_xdmf(file: str, field: str = "f", value_type: str = "double") -> df.MeshFunction:
        """
        Reads a meshfunction from a mapped field in a xdmf file.

        *Arguments*:
            file: String of input file
            field: String, identifier in xdmf file, default: "f"
            value_type: String of type of mapped field, default is double

        *Example*:
            mesh_function = read_mapped_xdmf("geometry.xdmf")
        """
        mesh = df.Mesh()
        file = df.XDMFFile(file)
        file.read(mesh)
        file.close()
        mvc = df.MeshValueCollection(value_type, mesh, mesh.topology().dim())
        with file as infile:
            infile.read(mvc, field)
        return df.MeshFunction(value_type, mesh, mvc)

    def map_field(self, field_file: str, mesh: Union[df.Mesh, str] = None, outfile: str = None) -> str:
        """
        Maps field onto mesh file.

        *Arguments*:
            field_file: Nifti file of field
            outfile:    String of output file
            mesh:       Dolfin mesh or path to mesh file

        *Example*:
            xdmf_file = map_field("edema.nii.gz", "edema", mesh)
        """
        if mesh is None:
            mesh = self.dolfin_mesh
        image = nib.load(field_file)
        data = image.get_fdata()

        if outfile is None:
            path, _, name = io.get_path_file_extension(field_file)
            outfile = os.path.join(path, name)


        if type(mesh) is str:
            mesh = FieldMapGenerator.load_mesh(mesh)

        n = mesh.topology().dim()
        regions = df.MeshFunction("double", mesh, n, 0)

        for cell in df.cells(mesh):
            c = cell.index()
            # Convert to voxel space
            ijk = cell.midpoint()[:]
            # Round off to nearest integers to find voxel indices
            i, j, k = np.rint(ijk).astype("int")
            # Insert image data into the mesh function:
            regions.array()[c] = float(data[i, j, k])

        # Store regions in XDMF
        xdmf = df.XDMFFile(mesh.mpi_comm(), outfile + ".xdmf")
        xdmf.parameters["flush_output"] = True
        xdmf.parameters["functions_share_mesh"] = True
        xdmf.write(mesh)
        xdmf.write(regions)
        xdmf.close()
        return outfile + ".xdmf"

    @staticmethod
    def image2array(image_dir: str) -> tuple[Any, Any, Any]:
        """
        Takes a directory of an image and gives a numpy array.

        *Arguments*:
            image_dir:      String of a Nifti image directory
        *Returns*:
            numpy array of image data, shape, affine
        """
        orig_image = nib.load(image_dir)
        return copy.deepcopy(orig_image.get_fdata()), orig_image.shape, orig_image.affine

    @staticmethod
    def image2mask(image_dir: str, compartment: int = None, inner_compartments: list[int] = None) -> np.ndarray:
        """
        Gives deep copy of original image with selected compartments.

        *Arguments*:
            image_dir:          String to Nifti image
            compartment:        Int, identifier of compartment that shall be filtered
            inner_compartments: List of inner compartments that also are included in the mask
        *Returns*:
            mask:               Numpy array of the binary mask
        """
        mask, _, _ = FieldMapGenerator.image2array(image_dir)
        unique = list(np.unique(mask))
        unique.remove(compartment)
        for outer in unique:
            mask[np.isclose(mask, outer)] = 0.0
        mask[np.isclose(mask, compartment)] = 1.0
        if inner_compartments is not None:
            for comp in inner_compartments:
                mask[np.isclose(mask, comp)] = 1.0
                unique.remove(comp)
        return mask

    def set_affine(self, image: Union[nib.Nifti1Image, str]) -> None:
        """
        Sets affine and shape of first measure of included state. The optional argument takes an nibabel Nifti1Image
        and takes the first measurement of the hold state of the mri entity if no argument is given. Affine and shape
        can be accessed via self.affine and self.shape.

        *Arguments*:
            image:      Optional nib.Nifti1Image, Default is self.state.measures[0].dir_act
        """
        if type(image) is str:
            image = nib.load(image)
        self.affine = image.affine
        self.shape = image.shape

    def interpolate(self, image, name: str, plateau=None, hole=None, min_value: float = 1.0,
                    max_value: float = 2.0, rest_value: float = 0.0, method: str = "linear") -> str:
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

        file_output = self.out_dir + name
        io.write_field2nii(values, file_output, self.affine)
        if not file_output.endswith("nii.gz"):
            file_output = file_output + "nii.gz"
        return file_output

    def run_brats(self, brats_seg: Union[str, list[str]], struc_seg: list[str]) -> None:
        """
        Collects masks of brats segmentation.
        Sets tumor classes analogous to the white and gray matter and csf. Needed for mean averaged value. List
        should have three entities. First for white matter, second for gray matter, third for csf.
        Maps white matter fields (white and grey and csf) onto geometry

        """
        # collect masks from segmentation file
        if type(brats_seg) is str:
            ede_mask, _, self.affine = FieldMapGenerator.image2mask(brats_seg, self.tumor_class_mapping["edema"])
            nec_mask = FieldMapGenerator.image2mask(brats_seg, self.tumor_class_mapping["necrotic"])
            act_mask = FieldMapGenerator.image2mask(brats_seg, self.tumor_class_mapping["active"])
        else:
            print("affine must be set by user!")
            ede_mask = brats_seg[0]
            act_mask = brats_seg[1]
            nec_mask = brats_seg[2]

        wm_mask = struc_seg[0]
        gm_mask = struc_seg[1]
        csf_seg = struc_seg[2]

        # edema mapping
        plateau = act_mask + nec_mask
        ede_ip = self.interpolate(ede_mask, "edema_ip", plateau=plateau,
                                  min_value=self.edema_min_value, max_value=self.edema_max_value,
                                  method=self.interpolation_method)
        self.mapped_ede_file = self.map_field(ede_ip, self.dolfin_mesh, self.out_dir + "edema")

        # solid tumor mapping
        # Needed to change edema with necrotic...somehow lead to overwriting of edema
        # generate separated nii maps
        act_ip = self.interpolate(act_mask, "active_ip", hole=nec_mask,
                                  min_value=self.active_min_value, max_value=self.active_max_value,
                                  method=self.interpolation_method)

        # hotfix, necrotic image has not nicely convex hull
        max_bound = find_boundaries((nec_mask + act_mask).astype(int), mode="inner")
        solid_tumor = nib.Nifti1Image(nec_mask + act_mask - max_bound, self.affine)
        active_tumor = nib.load(act_ip)
        nec = self.get_nec_image(active_tumor_img=active_tumor, solid_tumor_img=solid_tumor)
        # fsl.wrappers.fslmaths(active_tumor).div(active_tumor).mul(-1).add(solid_tumor).run()
        nib.save(nec, self.out_dir + "necrotic_ip.nii.gz")

        self.mapped_act_file = self.map_field(act_ip, self.dolfin_mesh, self.out_dir + "active")
        self.mapped_nec_file = self.map_field(self.out_dir + "necrotic_ip.nii.gz", self.dolfin_mesh,
                                              self.out_dir + "necrotic")

        # set mixed masks
        if self.structure_mapping_method == "const_wm":
            tumor_mask = nib.Nifti1Image(act_mask + nec_mask + ede_mask, self.affine)
            self.add_masks_and_save(wm_mask, tumor_mask, self.out_dir + "wm.nii.gz")
            self.mixed_wm_mask = self.out_dir + "wm.nii.gz"
            self.mixed_gm_mask = self.mri.gm_mask
            self.mixed_csf_mask = self.mri.csf_mask

        elif self.structure_mapping_method == "mean_averaged_value":
            self.add_masks_and_save(self.mri.wm_mask, classes[0], self.out_dir + "wm.nii.gz")
            self.add_masks_and_save(self.mri.gm_mask, classes[1], self.out_dir + "gm.nii.gz")
            self.add_masks_and_save(self.mri.csf_mask, classes[2], self.out_dir + "csf.nii.gz")
            self.mixed_wm_mask = self.out_dir + "wm.nii.gz"
            self.mixed_gm_mask = self.out_dir + "gm.nii.gz"
            self.mixed_csf_mask = self.out_dir + "csf.nii.gz"

        elif self.structure_mapping_method == "tumor_entity_weighted":
            print("not implemented")
            pass

        self.mapped_wm_file = self.map_field(self.mixed_wm_mask, self.xdmf_file, self.out_dir + "white_matter")
        self.mapped_gm_file = self.map_field(self.mixed_gm_mask, self.xdmf_file, self.out_dir + "gray_matter")
        self.mapped_csf_file = self.map_field(self.mixed_csf_mask, self.xdmf_file, self.out_dir + "csf")

    def get_nec_image(self, active_tumor_img: nib.Nifti1Image, solid_tumor_img: nib.Nifti1Image) -> nib.Nifti1Image:
        """
        Isolates the necrotic core from the solid tumor with the active outer rim.

        :param active_tumor_img: Active outer rim of solid tumor. nib.NiftiImage
        :param solid_tumor_img: Total solid tumor. nib.NiftiImage
        :return: nib.NiftiImage of the necrotic inner core
        """

        active_tumor_data = active_tumor_img.get_fdata()
        solid_tumor_data = solid_tumor_img.get_fdata()
        # Prevent division by zero by using np.where to avoid NaNs
        division_result = np.where(active_tumor_data != 0, 1, 0)
        nec_data = solid_tumor_data - division_result
        return nib.Nifti1Image(nec_data, active_tumor_img.affine, active_tumor_img.header)

    def add_masks_and_save(self, mask_one_img: nib.Nifti1Image, mask_two_img: Union[nib.Nifti1Image, np.ndarray],
                           output_path: str) -> None:
        """
        Adds two masks and saves the result.

        *Arguments*:
            wm_mask_path:   String path to the white matter mask NIFTI image
            tumor_mask_path: String path to the tumor mask NIFTI image
            output_path:    String path for the output NIFTI image
        """
        mask_one_data = mask_one_img.get_fdata()
        if type(mask_two_img) is nib.Nifti1Image:
            mask_two_data = mask_two_img.get_fdata()
        elif type(mask_two_img) is np.ndarray:
            mask_two_data = mask_two_img
        else:
            print("ERROR: mask two is not an nifti or an array")
            return None
        added_data = np.add(mask_one_data, mask_two_data)
        result_img = nib.Nifti1Image(added_data, mask_one_img.affine, mask_one_img.header)
        nib.save(result_img, output_path)
