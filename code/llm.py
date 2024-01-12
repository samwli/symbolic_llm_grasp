import openai
import re
import cv2
import ast 
import numpy as np
import torch
import torch.nn.functional as F
import os
from src.symbolic_llm_grasp.code.keys import API_KEY, API_ORG

# openai.api_base = "http://localhost:23002/v1"
def callOpenAI(api_key, model, query, organization=None):
    openai.api_key = api_key
    if organization:
        openai.organization = organization

    msg = [{"role": "user", "content": query}]
    try:
        response = openai.ChatCompletion.create(model=model, messages=msg)
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print("Skipping prediction:", str(e))
        return None

def read_graph_from_file(file_path):
    with open(file_path, 'r') as file:
        graph_data = file.read()
    return graph_data

def write_response_to_file(file_path, response):
    with open(file_path, 'w') as file:
        file.write(response)
        
def parse_nodes(graph_text):
        nodes_str = re.search(r"Nodes of the graph:\s*(\[.*\])", graph_text).group(1)
        nodes = ast.literal_eval(nodes_str)
        prompt = ""
        for node in nodes:
            node_name = node[0]
            prompt += f"{node_name}: __\n"

        return prompt
    
def parse_likelihoods(response):
    pattern = r"(rect\d+|ellip\d+|tri\d+|circ\d+)\s*[:\-]*\s*([0-9]*\.?[0-9]+)"
    matches = re.findall(pattern, response)
    
    return matches

def parse_graph_nodes(graph_text):
    nodes_str = re.search(r"Nodes of the graph:\s*(\[.*\])", graph_text).group(1)
    nodes = ast.literal_eval(nodes_str)
    node_data = {node[0]: (node[1]['angle'],) + node[1]['centroid'] for node in nodes}
    return node_data

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
    return data

def run_llm(graph_data, img, output_dir, obj_data_path, mode): 
    # graph_data = read_graph_from_file('outputs/controller_3d_20231226_152016/controller_graph.txt')
    input_obj = output_dir.split('/')[1].split('_'+mode)[0]
    give_object = True
    query = "Given the decomposition of object "
    if give_object:
        query += input_obj
        query += " "
    query += "into convex parts, each represented as a node in a graph, estimate the likelihood (between 0 and 1) that each part is the 'proper' part to grasp the object, like a handle or stem (give each part a different likelihood value so they can be ranked). Imagine you are a robot hand and want to pick the object up in a stable and safe manner. Consider the shape, centroid, area, aspect ratio, angle, color"
    if mode == "3d":
        query += ", and height"
    context = ". For example, graspable stems in flowers may be green, not-graspable metal in tools may be silver. Large area rectangles or large aspect ratio ellipses may (but not necessarily) represent handles and be more suitable than circles or shapes with small aspect ratio"
    give_context = True
    if give_context:
        query += context
    query_graph = ". The angles are measured counter-clockwise from the x-axis. Edges represent parts that share a boundary in the object and include the length of the shared boundary. \n" + graph_data + "\nRespond by filling in the following likelihood values for each node. Be succinct, return only the template with filled out values and nothing else. \n"
    query += query_graph
    prompt = parse_nodes(graph_data)
    query += prompt
    # openai.api_base = "http://localhost:23002/v1"
    response = callOpenAI(api_key=API_KEY, model="gpt-4-1106-preview", query=query, organization=API_ORG)
    likelihoods = parse_likelihoods(response)
    likelihoods = torch.tensor([float(match[1]) for match in likelihoods])
    
    most_likely_index = torch.argmax(likelihoods)
    node_data = parse_graph_nodes(graph_data)
    vis_file_path = output_dir+f"/llm_{input_obj}_grasp.png"
    # img = cv2.imread(output_dir+f"/{input_obj}_graph_shapes.png")
    most_likely_node = list(node_data.keys())[most_likely_index.item()]
    grasp_pose = node_data[most_likely_node]
    cv2.circle(img, grasp_pose[1:], radius=5, color=(0, 0, 255), thickness=-1)
    cv2.imwrite(vis_file_path, img)

    height_array = load_height(obj_data_path+'_depth')
    grasp_pose = np.append(grasp_pose, height_array[grasp_pose[2], grasp_pose[1]])
    
    print(f"Predicted Node: {most_likely_node}, Angle: {grasp_pose[0]}, Centroid: {grasp_pose[1:]}")

    output_file_path = output_dir+f"/llm_{input_obj}_results.txt"
    with open(output_file_path, 'w') as file:
        file.write("Response:\n" + response)
        file.write("\nLikelihoods:\n" + str(likelihoods.tolist()))
        file.write(f"\nPredicted Node: {most_likely_node}, Angle: {grasp_pose[0]}, Centroid: {grasp_pose[1:]}\n")
        
    return grasp_pose, img
