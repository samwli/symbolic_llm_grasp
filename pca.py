import numpy as np
from sklearn.decomposition import PCA
# import open3d as o3d
import pickle
import cv2
from calc3d import get_3d
import json

def parse_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        most_likely_index = None
        centroid = None
        for i, line in enumerate(lines):
            if "Likelihoods assignment response:" in line:
                # The next line contains the JSON string
                likelihoods_str = lines[i + 1].strip()
                try:
                    likelihoods_dict = json.loads(likelihoods_str)
                    # Convert values to float if they are not already
                    likelihoods_dict = {k: float(v) for k, v in likelihoods_dict.items()}
                    likelihoods = np.array(list(likelihoods_dict.values()))
                    most_likely_index = np.argmax(likelihoods)
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON: {e}")
                    return None, None
            elif "Centroid:" in line:
                # The line contains the centroid information, extract it
                centroid_str = line.split("Centroid:")[1].strip()
                centroid = np.fromstring(centroid_str[1:-1], dtype=int, sep=' ')
        return most_likely_index, centroid

def apply_pca_to_points(points_xyz):
    """
    Apply PCA on 3D points and return the principal components.

    Parameters:
    - points_xyz: N x 3 numpy array of 3D points.

    Returns:
    - Principal components (PCs).
    """
    pca = PCA(n_components=3)
    pca.fit(points_xyz)
    return pca.components_

# Load the 3D points (with XYZRGB) from a .npy file
obj = 'hammer'
task_iter = '1'
mode = '3d'
depthFrame = np.load(f'data/{obj}{task_iter}/{obj}_depth.npy')
image_arr = cv2.imread(f'data/{obj}{task_iter}/{obj}_rgb.png')
pkl_file = f'outputs_final/{obj}{task_iter}_{mode}_object/{obj}_2d_hulls.pkl'
with open(pkl_file, 'rb') as f:
    hulls = pickle.load(f)
 
file_path = f'outputs_final/{obj}{task_iter}_{mode}_object/llm_{obj}_results.txt'
hull_idx, centroid = parse_file(file_path)

hull = hulls[hull_idx]
image = np.zeros_like(depthFrame)
for simplex in hull.simplices:
    start_point = tuple(map(int, hull.points[simplex[0]]))  # Convert to tuple of ints
    end_point = tuple(map(int, hull.points[simplex[1]]))    # Convert to tuple of ints
    cv2.line(image, start_point, end_point, color=1)
binary_mask = np.zeros_like(image)
cv2.fillConvexPoly(binary_mask, np.array(hull.points[hull.vertices], dtype=np.int32), 1)

points_with_colors = get_3d(binary_mask, image_arr, depthFrame, obj)

# Separate the XYZ coordinates and RGB values
points_xyz = points_with_colors[:, :3]  # First three columns for XYZ
points_xyz -= points_xyz.mean(axis=0)
points_xyz /= 1000
# colors_rgb = points_with_colors[:, 3:6] / 255.0  # Last three columns for RGB, normalized to [0, 1]

# Apply PCA to the XYZ coordinates
PCs = apply_pca_to_points(points_xyz)

# Calculate the rotation in the XY plane (angle between PC1 and the X-axis)
angle_radians_xy = np.arctan2(PCs[0][1], PCs[0][0])
angle_degrees_xy = np.degrees(angle_radians_xy) % 360  # Ensure the angle is within [0, 360) degrees

# Calculate the rotation from the horizontal (angle between PC1 and the XY plane)
# This can be computed as the arcsin of the Z component of the normalized first principal component
PC1_normalized = PCs[0] / np.linalg.norm(PCs[0])
angle_from_horizontal_radians = np.arcsin(PC1_normalized[2])
angle_from_horizontal_degrees = np.degrees(angle_from_horizontal_radians)

grasp_pose = [angle_degrees_xy % 180, angle_from_horizontal_degrees % 180, centroid]
# Flatten the array
flattened = np.hstack(grasp_pose)

# Convert the elements to strings and join them with spaces
grasp_pose_str = ' '.join(map(str, flattened))

# Print the result
print(grasp_pose_str)
# # Visualization (optional)
# # Note: This part visualizes the point cloud with Open3D, but without PCA vectors
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points_xyz)
# # Optional: Assign colors if you wish to visualize the RGB data
# # pcd.colors = o3d.utility.Vector3dVector(colors_rgb)
# o3d.visualization.draw_geometries([pcd])