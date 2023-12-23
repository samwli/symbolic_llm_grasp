import os
import argparse
import datetime
import numpy as np
from code.mesh import create_solid_mesh
from code.decompose import decompose
from code.graph import create_graph
from code.llm import run_llm

def create_output_directory(obj, mode):
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join('outputs', f'{obj}_{mode}_{current_time}')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def run_pipeline(obj, data_dir, mode):
    obj_data_path = os.path.join(data_dir, obj)
    output_dir = create_output_directory(obj, mode)
    mesh_file = os.path.join(output_dir, f'{obj}_solid_mesh.obj')
    create_solid_mesh(obj_data_path, mesh_file, mode)
    threshold = 0.1 if mode == '3d' else 0.07
    num_hulls = decompose(output_dir, obj_data_path, obj, mode, threshold)
    while num_hulls < 2 and threshold > 0.01:
        threshold = max(threshold - 0.02, 0.01)
        num_hulls = decompose(output_dir, obj_data_path, obj, mode, threshold)
    create_graph(output_dir, obj_data_path, obj, mode)
    grasp_point = np.array(run_llm(output_dir, obj, mode))
    if mode == '3d':
        height_array = np.load(obj_data_path+'_height.npy')
        grasp_point = np.append(grasp_point, height_array[grasp_point[1], grasp_point[0]])
        print(f"3D grasp point: {grasp_point}")
    # TODO: grasp at this point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the 3D mesh pipeline')
    parser.add_argument('--obj', type=str, default='knife', help='Input object identifier')
    parser.add_argument('--data_dir', type=str, default='data/', help='Directory where object data is located')
    parser.add_argument('--mode', type=str, default='3d', help='3d or 2d (use depth or no depth)')
    args = parser.parse_args()

    run_pipeline(args.obj, args.data_dir, args.mode)
