import numpy as np
import coacd
import trimesh
from scipy.spatial import ConvexHull
from PIL import Image, ImageDraw
import pickle

def draw_hulls(img, hulls, output_dir, obj):
    if img.mode == 'L':
        img = img.convert('RGB')
    draw = ImageDraw.Draw(img)
    
    for hull in hulls:
        for simplex in hull.simplices:
            points = [tuple(hull.points[i]) for i in simplex]
            draw.line(points, fill='red', width=2)
    img.save(output_dir+f'/{obj}_2d_hulls.png')   


def decompose(output_dir, obj_data_path, obj, threshold = 0.08):
    mesh = trimesh.load(output_dir+f'/{obj}_solid_mesh.obj', force="mesh")
    coacd.set_log_level("error")
    mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    result = coacd.run_coacd(
        mesh,
        threshold=threshold
    )

    mesh_parts = []
    for vs, fs in result:
        mesh_parts.append(trimesh.Trimesh(vs, fs))
    scene = trimesh.Scene()
    np.random.seed(0)
    for p in mesh_parts:
        p.visual.vertex_colors[:, :3] = (np.random.rand(3) * 255).astype(np.uint8)
        scene.add_geometry(p)
    scene.export(output_dir+f'/{obj}_convex_parts.obj')

    vs_list = [vs for vs, fs in result]
    filtered_vs_list = [vs[:, :2] for vs in vs_list]
    
    rotate_matrix = np.array([[0, 1], [-1, 0]])
    flip_matrix = np.array([[-1, 0], [0, 1]])
    
    hulls = []
    for array_2d in filtered_vs_list:
        if len(array_2d) > 2: 
            array_2d = np.dot(array_2d, rotate_matrix) 
            array_2d = np.dot(array_2d, flip_matrix)  
            hull = ConvexHull(array_2d)
            hulls.append(hull)

    with open(output_dir+f'/{obj}_2d_hulls.pkl', 'wb') as f:
        pickle.dump(hulls, f)
    

    rgb_image = Image.open(obj_data_path+'_rgb.png')
    draw_hulls(rgb_image, hulls, output_dir, obj)
    
