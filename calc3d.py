import numpy as np
from code.load_data import load_mask, load_height
import cv2
import math
import os

def calculate3D(depthFrame, points):
    points3d = []
    for point in points:
        
        p3d = calc_spatial_point(depthFrame, point[:2])
        p3d = [p3d['x'], p3d['y'], p3d['z'], point[2], point[3], point[4]]
        # If there is no nan value in p3d, add it:
        if not np.isnan(p3d).any():
            points3d.append(p3d)
    return np.asarray(points3d)

def calc_spatial_point(depthFrame, point):
    centroid = {'x': point[0], 'y': point[1]}
    depth = depthFrame[centroid['y'], centroid['x']]
    HFOV = 1.2541936111357697
    midW = int(depthFrame.shape[1] / 2)
    midH = int(depthFrame.shape[0] / 2)
    bb_x_pos = centroid['x'] - midW
    bb_y_pos = centroid['y'] - midH
    angle_x = calc_angle(depthFrame, bb_x_pos, HFOV)
    angle_y = calc_angle(depthFrame, bb_y_pos, HFOV)
    spatials = {
        'z': depth,
        'x': depth * math.tan(angle_x),
        'y': -1 * depth * math.tan(angle_y)
    }
    return spatials

def calc_angle(frame, offset, HFOV):
    return math.atan(math.tan(HFOV / 2.0) * offset / (frame.shape[1] / 2.0))

objs = [name[:-1] for name in os.listdir('data')]
# mask = cv2.imread(f'data/{obj}1/{obj}_mask.png')
def get_3d(mask, image_arr, depthFrame, obj):
    # mask = load_mask(f'data/{obj}1/{obj}_mask')
    # image_arr = cv2.imread(f'data/{obj}1/{obj}_rgb.png')
    # depthFrame = load_height(f'data/{obj}1/{obj}_depth')
    points = []
    for y in range(0, mask.shape[0], 1):
        for x in range(0, mask.shape[1], 1):
            if mask[y,x] > 0:
                r,g,b = image_arr[y,x,:]
                points.append([x,y,r,g,b])

    points_3d = calculate3D(depthFrame, points)
    return points_3d
    # np.save(f'pc_data/{obj}_points_3d.npy', points_3d)
    # np.save(f'data/{obj}1/{obj}_points_3d.npy', points_3d)
    # print("calc3d points:")
    # print(points_3d)

    # print("load 3d points:")
    # load_points = np.load(f'data/{obj}1/{obj}_points_3d.npy')
    # print(load_points)

    # print(np.array_equal(points_3d, load_points))   