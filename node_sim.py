import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import pandas as pd

def compute_node_similarity_matrix(graph):
    nodes_list = list(graph.nodes())
    n = len(nodes_list)

    # Initialize a similarity matrix
    similarity_matrix = np.zeros((n, n))

    # Compute the Jaccard similarity coefficient for all pairs of nodes
    for i, node_u in enumerate(tqdm(nodes_list, desc="Computing node similarities")):
        neighbors_u = set(graph.neighbors(node_u))
        for j, node_v in enumerate(nodes_list):
            if i != j:
                neighbors_v = set(graph.neighbors(node_v))
                intersection = len(neighbors_u & neighbors_v)
                union = len(neighbors_u | neighbors_v)
                similarity_matrix[i, j] = intersection / union if union != 0 else 0

    return similarity_matrix, nodes_list


def detect_communities(graph):
    num_communities = 30
    similarity_matrix, nodes_list = compute_node_similarity_matrix(graph)

    # Apply KMeans clustering on the similarity matrix
    kmeans = KMeans(n_clusters=num_communities)
    clusters = kmeans.fit_predict(similarity_matrix)

    # Create a dictionary to store the nodes in each community
    communities = {i: [] for i in range(num_communities)}
    for node, cluster in zip(nodes_list, clusters):
        communities[cluster].append(node)

    # print(communities)

    community_df = pd.DataFrame(communities.items(), columns=['Community_id', 'Nodes'])
    community_df['len'] = community_df['Nodes'].apply(len)
    # community_df = community_df.reset_index()
    # community_df = community_df.rename(columns={'index': 'Community_id'})
    print(community_df)
    return community_df