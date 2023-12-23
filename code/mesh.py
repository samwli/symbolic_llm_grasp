import os
import numpy as np
from trimesh import voxel

def create_solid_mesh(obj_data_path, mesh_file, mode):
    if mode == '3d':
        mask = np.load(obj_data_path+'_mask.npy')
        height_array = np.load(obj_data_path+'_height.npy')
        normalized_height = np.clip(height_array / height_array.max(), 0, 1)
        min_depth_inside_mask = normalized_height[mask].min()
        normalized_height -= min_depth_inside_mask - 0.011
        normalized_height = np.clip(normalized_height, 0, 1) 
        voxels = np.zeros((mask.shape[0], mask.shape[1], int(normalized_height.max() * 100)), dtype=bool)
        for i in range(voxels.shape[0]):
            for j in range(voxels.shape[1]):
                if not np.all(mask[i, j] == 0): 
                    height = int(normalized_height[i, j] * 100)
                    voxels[i, j, :height] = True
        solid_mesh = voxel.ops.matrix_to_marching_cubes(voxels, pitch=1)
        solid_mesh.export(mesh_file)
    elif mode == '2d':
        mask = np.load(obj_data_path+'_mask.npy')
        # process real world mask if needed
        solid_mesh = voxel.ops.matrix_to_marching_cubes([mask], pitch=1)
        solid_mesh.export(mesh_file)