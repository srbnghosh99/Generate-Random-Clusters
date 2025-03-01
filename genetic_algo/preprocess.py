import pandas as pd
import ast

def process_solutions(filename,G):
    df = pd.read_csv(filename)
    nodearray = G.nodes()

    node_to_community = {}
    df['Nodes'] = df['Nodes'].apply(ast.literal_eval)
    # print(df)
    for _, row in df.iterrows():
        for node in row["Nodes"]:
            node_to_community[node] = row["Community_id"]

    # Map nodelist to community IDs
    solution = [node_to_community.get(node, None) for node in nodearray]  # None for missing nodes

    # print(mapped_communities)
    return solution
