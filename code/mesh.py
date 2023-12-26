import numpy as np
import os
from trimesh import voxel
import cv2

def load_height(base_path):
    if os.path.exists(base_path + '.npy'):
        data = np.load(base_path + '.npy')
    elif os.path.exists(base_path + '.png'):
        image = cv2.imread(base_path + '.png')
        data = np.array(image)
    else:
        raise ValueError("File not found or unsupported file format (npy or png)")
    
    if len(data.shape) > 2:
        data = data[:, :, 0]
        
    data = cv2.medianBlur(data, 5)
    data = cv2.GaussianBlur(data, (5, 5), 0)
    
    return data

def load_mask(base_path):
    if os.path.exists(base_path + '.npy'):
        mask = np.load(base_path + '.npy')
    elif os.path.exists(base_path + '.png'):
        mask = cv2.imread(base_path + '.png')
        mask = np.array(mask)
    else:
        raise ValueError("File not found or unsupported file format (npy or png)")
    
    if len(mask.shape) > 2: 
        mask = np.any(mask > 0, axis=2).astype(np.uint8)
    mask = np.where(mask > 0, 1, 0).astype(np.uint8)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    mask_largest_contour = np.zeros_like(mask)
    cv2.drawContours(mask_largest_contour, [largest_contour], -1, color=255, thickness=cv2.FILLED)
    return np.where(mask_largest_contour > 0, 1, 0)


def create_solid_mesh(obj_data_path, mesh_file, mode):
    mask = load_mask(obj_data_path+'_mask')
    if mode == '3d':
        height_array = load_height(obj_data_path+'_depth')
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
    elif mode == '2d':
        # process real world mask if needed
        solid_mesh = voxel.ops.matrix_to_marching_cubes([mask], pitch=1)
    # solid_mesh.export(mesh_file)
    return solid_mesh

