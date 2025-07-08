import pandas as pd
import ast
import networkx as nx
import random

def process_solutions(filename,G):
    print(filename)
    df = pd.read_csv(filename)
    nodearray = G.nodes()
    node_to_community = {}
    df['Nodes'] = df['Nodes'].apply(ast.literal_eval)
    for _, row in df.iterrows():
        for node in row["Nodes"]:
            node_to_community[node] = row["Community_id"]
    solution = []
    Community_id_list = df['Community_id'].tolist()
    for node in nodearray:
        if node not in node_to_community:
            random_id = random.choice(Community_id_list)
            node_to_community[node] = random_id
            solution.append(node_to_community[node])
        else:
            solution.append(node_to_community[node])
    return solution


