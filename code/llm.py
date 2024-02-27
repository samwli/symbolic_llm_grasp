import openai
import re
import cv2
import ast 
import numpy as np
import os
from scipy.spatial.distance import cdist
from code.keys import API_KEY, API_ORG
from code.load_data import load_height, load_mask
from code.typechat.typechat import TypeChat
import json

def generate_ts_schema_for_likelihoods(nodes):
    # Start the interface definition
    schema = "// The following is a schema definition for assigning a single likelihood (value) to each node (key).\n\n"
    schema += "export interface NodeLikelihoods {\n"
    # For each node, add a property to the schema
    for node in nodes:
        schema += f"    {node}: number; // put here only the likelihood between 0 and 1 that the robot should grasp {node} and nothing else.\n"
    # Close the interface
    schema += "}"
    return 'NodeLikelihoods', schema

def generate_ts_schema_for_parts(nodes):
    # Start the interface definition
    schema = "// The following is a schema definition for assigning a single semantic part (value) to each node (key).\n\n"
    schema = "export interface NodePartsAssignment {\n"
    # For each node, add a property to the schema
    for node in nodes:
        schema += f"    {node}: string;  // put here only the semantic part assigned to {node} as a string and nothing else.\n" 
    # Close the interface
    schema += "}"
    return 'NodePartsAssignment', schema

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
        
# def parse_nodes(graph_text):
#         nodes_str = re.search(r"Nodes of the graph:\s*(\[.*\])", graph_text).group(1)
#         nodes = ast.literal_eval(nodes_str)
#         prompt = ""
#         for node in nodes:
#             node_name = node[0]
#             prompt += f"{node_name}: __\n"

#         return prompt

def parse_nodes(graph_text):
        nodes_str = re.search(r"Nodes of the graph:\s*(\[.*\])", graph_text).group(1)
        nodes = ast.literal_eval(nodes_str)
        node_names = [node[0] for node in nodes]

        return node_names
    
def parse_likelihoods(response):
    pattern = r"(rect\d+|ellip\d+|tri\d+|circ\d+)\s*[:\-]*\s*([0-9]*\.?[0-9]+)"
    matches = re.findall(pattern, response)
    
    return matches

def parse_graph_nodes(graph_text):
    nodes_str = re.search(r"Nodes of the graph:\s*(\[.*\])", graph_text).group(1)
    nodes = ast.literal_eval(nodes_str)
    node_data = {node[0]: (node[1]['angle'],) + node[1]['centroid'] for node in nodes}
    return node_data

def run_llm_old(graph_data, img, output_dir, obj_data_path, mode, no_object): 
    return
    # graph_data = read_graph_from_file('outputs/controller_3d_20231226_152016/controller_graph.txt')
    input_obj = output_dir.split('/')[1].split('_'+mode)[0][:-1]
    query = "Given the decomposition of "
    if not no_object:
        query += input_obj
        query += " "
    else:
        query += "an object "
    query += "into convex parts, each represented as a node in a graph, estimate the (distinct) likelihood between 0 and 1 that each part is the 'proper' part to grasp the object, like a handle or stem. Imagine you are a robot hand and want to pick the object up in a stable and safe manner. Consider the shape, position, color"
    if mode == "3d":
        query += ", and height"
    context = ". For example, large area and aspect ratio rectangles or ellipses may (but not necessarily) represent handles and be more suitable to grasp than circles or shapes with small area/aspect ratio. Metal parts in tools may be silver/gray and should be avoided"
    give_context = True
    if give_context:
        query += context
    query_graph = ". Edges represent parts that are connected in the object and include the length of the shared boundary.\n" + graph_data + "\nRespond by filling in the likelihood values for each node. Be succinct, return only the template with filled out values and nothing else. \n"
    query += query_graph
    prompt = parse_nodes(graph_data)
    query += prompt
    
    with open('tape_measure_prompt.txt', 'w') as f:
        f.write(query)
    
    # input_obj = output_dir.split('/')[1].split('_'+mode)[0][:-1]
    # query = "Given the decomposition of "
    # if not no_object:
    #     query += input_obj
    #     query += " "
    # else:
    #     query += "an object "
    # part = 'cone'
    # query += f"into convex parts, each represented as a node in a graph, estimate the (distinct) likelihood between 0 and 1 that each part is the {part}. Imagine you are a robot hand and want to pick the object up in a stable and safe manner. Consider the shape, position, color"
    # if mode == "3d":
    #     query += ", and height"
    # context = ". For example, large area or aspect ratio rectangles and ellipses may (but not necessarily) represent handles and be more suitable to grasp than circles or shapes with small area/aspect ratio. Metal blades in tools may be silver/gray and should be avoided"
    # give_context = True
    # if give_context:
    #     query += context
    # query_graph = ". Edges represent parts that are connected in the object and include the length of the shared boundary.\n" + graph_data + "\nRespond by filling in the likelihood values for each node. Be succinct, return only the template with filled out values and nothing else. \n"
    # query += query_graph
    # prompt = parse_nodes(graph_data)
    # query += prompt
    
    # query = "Given the decomposition of object "
    # if not no_object:
    #     query += input_obj
    #     query += " "
    # query += "into convex parts, each represented as a node in a graph, estimate the (distinct) likelihood between 0 and 1 that each part is transparent. Imagine you are a robot hand and want to pick the object up in a stable and safe manner. Consider the shape, position, color"
    # if mode == "3d":
    #     query += ", and height"
    # context = ". For example, large area or aspect ratio rectangles and ellipses may be more likely to represent handles than circles or shapes with small area/aspect ratio. Metal blades in tools may be silver/gray"
    # give_context = True
    # if give_context:
    #     query += context
    # query_graph = ". Edges represent parts that are connected in the object and include the length of the shared boundary. \n" + graph_data + "\nRespond by filling in the following likelihood values for each node. Be succinct, return only the template with filled out values and nothing else. \n"
    # query += query_graph
    # prompt = parse_nodes(graph_data)
    # query += prompt
    
    # openai.api_base = "http://localhost:23002/v1"
    response = callOpenAI(api_key=API_KEY, model="gpt-4-1106-preview", query=query, organization=API_ORG)
    likelihoods = parse_likelihoods(response)
    likelihoods = np.array([float(match[1]) for match in likelihoods])
    
    most_likely_index = np.argmax(likelihoods)
    node_data = parse_graph_nodes(graph_data)
    vis_file_path = output_dir+f"/llm_{input_obj}_grasp.png"
    # img = cv2.imread(output_dir+f"/{input_obj}_graph_shapes.png")
    most_likely_node = list(node_data.keys())[most_likely_index.item()]
    grasp_pose = node_data[most_likely_node]
    # TODO
    mask = load_mask(obj_data_path+'_mask')
    if mask[grasp_pose[1:][::-1]] == 0:
        object_indices = np.argwhere(mask == 1)
        distances = cdist(object_indices, [grasp_pose[1:][::-1]])
        min_index = np.argmin(distances)
        closest_point = object_indices[min_index]
        grasp_pose = list(grasp_pose)
        grasp_pose[1:] = closest_point[::-1]
    cv2.circle(img, grasp_pose[1:], radius=5, color=(0, 0, 255), thickness=-1)
    cv2.imwrite(vis_file_path, img)

    height_array = load_height(obj_data_path+'_depth')
    grasp_pose = np.append(grasp_pose, height_array[grasp_pose[2], grasp_pose[1]])
    
    print(f"Predicted Node: {most_likely_node}, Angle: {int(grasp_pose[0])}, Centroid: {grasp_pose[1:].astype(int)}")
    print(grasp_pose.astype(int).tolist())

    output_file_path = output_dir+f"/llm_{input_obj}_results.txt"
    with open(output_file_path, 'w') as file:
        file.write("Response:\n" + response)
        file.write("\nLikelihoods:\n" + str(likelihoods.tolist()))
        file.write(f"\nPredicted Node: {most_likely_node}, Angle: {int(grasp_pose[0])}, Centroid: {grasp_pose[1:].astype(int)}\n")
        
    return grasp_pose, img

def run_llm(graph, img, output_dir, obj_data_path, mode, no_object, task_string, model='gpt4'):
    ts = TypeChat()
    # Set up the language model that you want to use
    # Note: If you are utilizing local LLMs (e.g. through FastChat), you can set base_url to the URL of your local LLM
    # Note: You can enable json_mode if you want, however, only gpt-4-1106-preview supports this
    if model == 'gpt4':
        ts.createLanguageModel(model="gpt-4-0125-preview", api_key=API_KEY, org_key=API_ORG, use_json_mode=True)
    else:
        ts.createLanguageModel(model="Starling-LM-7B-alpha", api_key=API_KEY, org_key=API_ORG, base_url="http://localhost:23002/v1")
    obj = output_dir.split('/')[1].split('_'+mode)[0][:-1]
    obj_name = obj.replace("_", " ")
    nodes = parse_nodes(graph)
    output_file_path = output_dir+f"/llm_{obj}_results.txt"
    # parts reasoning
    parts_reasoning_schema = "./code/typechat/schemas/partsReasoningSchema.ts"
    ts.loadSchema(parts_reasoning_schema)
    parts_reasoning_name = "PartsReasoning"
    tns = ts.createJsonTranslator(name=parts_reasoning_name)
    if not no_object:
        request = [
            {"role": "user", "content": f"Given an object '{obj_name}' viewed from the top down, decomposed into convex parts each represented as a node in a graph, and task '{task_string}'. These nodes have attributes including shape, aspect ratio (defined as the length of the long axis over the short axis), color, and area. Edges in the graph represent physical connections between these parts within the object.\n\n"},
        ]
    else:
        request = [
            {"role": "user", "content": f"Given an object viewed from the top down, decomposed into convex parts each represented as a node in a graph, and task '{task_string}'. These nodes have attributes including shape, aspect ratio (defined as the length of the long axis over the short axis), color, and area. Edges in the graph represent physical connections between these parts within the object.\n\n"},
        ]
    request[0]["content"] += graph 
    request[0]["content"] += "\n\nConsider the main semantic and geometric parts the object may be decomposed into. Reason individually over all nodes in the graph about what semantic part each node may represent. Be succinct, give only a short one sentence explanation for each node." 
    return_query = False
    response = tns.translate(request, image=None, return_query=return_query, free_form=True)
    if return_query:
        print(response)
    else:
        if response.success:
            parts_reasoning_response = response.data["reasoning"]
            with open(output_file_path, 'w') as file:
                file.write("Parts reasoning response:\n" + parts_reasoning_response)
        else:
            print("parts reasoning error:", response.message)

    # now assign semantic parts to each node
    parts_name, ts_schema_parts = generate_ts_schema_for_parts(nodes)
    ts.loadSchema(schema=ts_schema_parts)
    tns = ts.createJsonTranslator(name=parts_name)
    assign_parts_request = request + [
            {"role": "assistant", "content": parts_reasoning_response},
            {"role": "user", "content": f"Assign a semantic part to each node."},
        ]
    return_query = False
    response = tns.translate(assign_parts_request, image=None, return_query=return_query, free_form=True)

    if return_query:
        print(response)
    else:
        if response.success:
            with open(output_file_path, 'a') as file:
                parts_assignment = response.data
                file.write("\n\nPart assignment response:\n")
                json.dump(response.data, file)
        else:
            print("part assignment error:", response.message)

    likelihoods_reasoning_schema = "./code/typechat/schemas/likelihoodsReasoningSchema.ts"
    likelihoods_reasoning_name = "LikelihoodsReasoning"
    ts.loadSchema(likelihoods_reasoning_schema)
    tns = ts.createJsonTranslator(name=likelihoods_reasoning_name)
    task = f"Imagine you are a robot hand and tasked to '{task_string}' in a proper and safe manner by selecting a part that gives you appropriate control of the object/part of interest. Reason about how likely each node/part is correct for the gripper to interact with. Be succinct, give only a short one sentence explanation for each node."
    # task = f"Imagine you are a three-fingered robot hand with a maximum width of 4 cm. You are tasked to '{task_string}' in a proper and safe manner by selecting a part that gives you appropriate control of the object/part of interest. Reason about how likely each node/part is correct for the gripper to interact with. Be succinct, give only a short one sentence explanation for each node."
    # task = f"Imagine you are a robot hand tasked with picking up '{obj_name}' and handing it to a human in such a way that the human can easily and safely grip the proper part for usage with minimal obstruction. Identify the best part of the '{obj_name}' for the robot to grasp and provide a short explanation for your choice, considering the ease of handover and safety."
    # task = f"Imagine you are a robot hand tasked with picking up 'the hot {obj_name}' and handing it to a human in such a way that the human can easily and safely grip the proper part for usage with minimal obstruction. Identify the best part of the '{obj_name}' for the robot to grasp and provide a short explanation for your choice, considering the ease of handover and safety."
    if response.success:
        request = assign_parts_request + [
            {"role": "assistant", "content": json.dumps(parts_assignment)},
            # {"role": "assistant", "content": parts_reasoning_response},
            {"role": "user", "content": task},
        ]
    else:
        request += [
            {"role": "assistant", "content": parts_reasoning_response},
            {"role": "user", "content": task},
        ]
        
    return_query = False
    response = tns.translate(request, image=None, return_query=return_query, free_form=True)
    if return_query:
        print(response)
    else:
        if response.success:
            # The response data is a dictionary with the keys being the names of the fields in your schema
            likelihoods_reasoning_response = response.data["reasoning"]
            with open(output_file_path, 'a') as file:
                file.write("\n\nLikelihoods reasoning response:\n" + likelihoods_reasoning_response)
        else:
            print("Error:", response.message)

    # now assign numeric likelihoods to each part
    likelihoods_name, ts_schema_likelihoods = generate_ts_schema_for_likelihoods(nodes)
    ts.loadSchema(schema=ts_schema_likelihoods)
    tns = ts.createJsonTranslator(name=likelihoods_name)
    # task = f"\n\nImagine you are a robot hand and want to lift the {obj} in a safe and secure manner. Give (distinct) likelihood estimates between 0 and 1 that each part is the correct part for the robot to grasp."
    # task = f"\n\nImagine you are a robot hand and want to pick up the {obj} and hand it to a human, in a way such that the human can receive and grasp it by a safe and secure part. Give (distinct) likelihood estimates between 0 and 1 that each part is the correct part for the robot to grasp."
    task = f"\n\nAssign a likelihood to each node for the task."
    request += [
        {"role": "assistant", "content": likelihoods_reasoning_response},
        {"role": "user", "content": task},
    ]
    return_query = False
    response = tns.translate(request, image=None, return_query=return_query, free_form=True)
    # The response object has a few fileds:
    # - success: True if the request was successful
    # - data: The response data
    # - error: The error message, if any
    if return_query:
        print(response)
    else:
        if response.success:
            # The response data is a dictionary with the keys being the names of the fields in your schema
            with open(output_file_path, 'a') as file:
                likelihoods = response.data
                file.write("\n\nLikelihoods assignment response:\n")
                json.dump(likelihoods, file)
        else:
            print("Error:", response.message)
    
    likelihoods = np.array(list(likelihoods.values()))
    most_likely_index = np.argmax(likelihoods)
    node_data = parse_graph_nodes(graph)
    vis_file_path = output_dir+f"/llm_{obj}_grasp.png"
    # img = cv2.imread(output_dir+f"/{input_obj}_graph_shapes.png")
    most_likely_node = list(node_data.keys())[most_likely_index.item()]
    grasp_pose = node_data[most_likely_node]
    # TODO
    mask = load_mask(obj_data_path+'_mask')
    if mask[grasp_pose[1:][::-1]] == 0:
        object_indices = np.argwhere(mask == 1)
        distances = cdist(object_indices, [grasp_pose[1:][::-1]])
        min_index = np.argmin(distances)
        closest_point = object_indices[min_index]
        grasp_pose = list(grasp_pose)
        grasp_pose[1:] = closest_point[::-1]
    cv2.circle(img, grasp_pose[1:], radius=5, color=(0, 0, 255), thickness=-1)
    cv2.imwrite(vis_file_path, img)

    height_array = load_height(obj_data_path+'_depth')
    grasp_pose = np.append(grasp_pose, height_array[grasp_pose[2], grasp_pose[1]])
    
    print(f"Predicted Node: {most_likely_node}, Angle: {int(grasp_pose[0])}, Centroid: {grasp_pose[1:].astype(int)}")
    print(' '.join(grasp_pose.astype(int).astype(str))+"\n")
    
    request_file_path = output_dir+f"/llm_{obj}_prompt.txt"
    with open(request_file_path, 'w') as file:
        for item in request:
            file.write("%s\n" % item)
    
    with open(output_file_path, 'a') as file:
        file.write(f"\n\nPredicted Node: {most_likely_node}, Angle: {int(grasp_pose[0])}, Centroid: {grasp_pose[1:].astype(int)}")
        
    return grasp_pose, img, most_likely_index