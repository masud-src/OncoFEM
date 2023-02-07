"""tumor map generator"""

from os import remove
from os.path import exists
import numpy as np
import nibabel as nib
from skimage.measure import regionprops
from oncofem.struc.study import Study
from oncofem.helper.general import mkdir_if_not_exist, run_shell_command, set_working_folder

class TumorMapGenerator:

    def __init__(self, study: Study, working_dir):
        self.study = study
        self.maps_dir = set_working_folder(working_dir + "tumor_maps/")
        self.labeled_image = None
        self.orig_field = None
        self.orig_affine = None
        self.out_zero_field = None
        self.labels = None
        self.props = None
        self.solid_tumor_nii = None
        self.necrotic_nii = None
        self.edema_nii = None
        self.max_edema_value = None
        self.max_solid_tumor_value = None
        self.max_necrotic_value = None

    def write_to_file(self, out, name):
        mkdir_if_not_exist(self.maps_dir)
        file = nib.Nifti1Image(out.astype(np.float64), self.orig_affine)
        nib.save(file, self.maps_dir + name)
        if exists(self.maps_dir + name + ".gz"):
            remove(self.maps_dir + name + ".gz")
        run_shell_command("gzip " + self.maps_dir + name)
        return self.maps_dir + name + ".gz"

    def read_labelprop_from_image(self, path_to_labeled_image: str):
        self.labeled_image = nib.load(path_to_labeled_image)
        self.orig_affine = self.labeled_image.affine
        # get python array with labels
        self.orig_field = self.labeled_image.get_fdata()
        self.out_zero_field = np.zeros(np.shape(self.orig_field))
        # get bitmask for desired labels
        # must be int type
        self.labels = self.orig_field.astype(int)
        # compute geometrical properties of bitmask
        self.props = regionprops(self.labels, self.orig_field)

    def generate_solid_tumor_map(self):
        out = self.out_zero_field
        voxels_tumor = self.props[2].coords
        for i, voxel in enumerate(voxels_tumor):
            out[voxel[0], voxel[1], voxel[2]] = self.max_solid_tumor_value #TODO: Fix this value

        self.solid_tumor_nii = self.write_to_file(out, "solid_tumor_map.nii")

    def generate_necrotic_tumor_map(self):
        out = self.out_zero_field
        voxels_necrotic = self.props[0].coords
        for i, voxel in enumerate(voxels_necrotic):
            out[voxel[0], voxel[1], voxel[2]] = self.max_necrotic_value #TODO: Fix this value

        self.necrotic_nii = self.write_to_file(out, "necrotic_core_map.nii")

    def generate_edema_map(self):
        necrotic_voxels = self.props[0].coords
        necrotic_centroid = self.props[0].centroid
        necrotic_area = self.props[0].area
        necrotic_weight = 1
        solid_tumor_voxels = self.props[2].coords
        solid_tumor_centroid = self.props[2].centroid
        solid_tumor_area = self.props[2].area
        solid_tumor_weight = 1
        edema_voxels = self.props[1].coords
        edema_centroid = self.props[1].centroid
        edema_area = self.props[1].area
        edema_weight = 1

        momentum_x = necrotic_centroid[0] * necrotic_weight * necrotic_area + solid_tumor_centroid[0] * solid_tumor_weight * solid_tumor_area + edema_centroid[0] * edema_weight * edema_area 
        momentum_y = necrotic_centroid[1] * necrotic_weight * necrotic_area + solid_tumor_centroid[1] * solid_tumor_weight * solid_tumor_area + edema_centroid[1] * edema_weight * edema_area 
        momentum_z = necrotic_centroid[2] * necrotic_weight * necrotic_area + solid_tumor_centroid[2] * solid_tumor_weight * solid_tumor_area + edema_centroid[2] * edema_weight * edema_area 
        weights = necrotic_weight * necrotic_area + solid_tumor_weight * solid_tumor_area + edema_weight * edema_area 

        overall_centroid = np.asarray([momentum_x / weights, momentum_y / weights, momentum_z / weights])
        dist_voxel_center_edema = [np.linalg.norm(voxel - overall_centroid) for voxel in edema_voxels]

        max_dist_voxel_edema = max(dist_voxel_center_edema)
        normed_dist = np.zeros((len(dist_voxel_center_edema), 1))
        for i, dist_voxel in enumerate(dist_voxel_center_edema):
            x = dist_voxel / max_dist_voxel_edema * np.pi
            normed_dist[i] = np.sqrt(2) / (0.798 * np.sqrt(np.pi)) * np.exp(-x * x / (0.798 * 0.798 * 2.0))

        out = self.out_zero_field
        for i, voxel in enumerate(edema_voxels):
            out[voxel[0], voxel[1], voxel[2]] = normed_dist[i] * self.max_edema_value #TODO: Fix this value

        #for i, voxel in enumerate(necrotic_voxels):
        #    out[voxel[0], voxel[1], voxel[2]] = 0

        #for i, voxel in enumerate(solid_tumor_voxels):
        #    out[voxel[0], voxel[1], voxel[2]] = 0

        self.edema_nii = self.write_to_file(out, "edema_map.nii")
