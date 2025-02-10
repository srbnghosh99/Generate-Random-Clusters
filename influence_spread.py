import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict

def influence_spread_communities(G):
    # G = nx.karate_club_graph()  # Example dataset
    # Compute node centrality (choose one: PageRank, Degree, or Betweenness)
    # centrality = nx.pagerank(G)  # Can replace with nx.betweenness_centrality(G) or nx.degree_centrality(G)
    centrality = nx.betweenness_centrality(G)

    # Sort nodes by centrality and select top-K as community seeds
    num_clusters = 30  # Define number of communities
    top_nodes = sorted(centrality, key=centrality.get, reverse=True)[:num_clusters]

    # Create a feature matrix using shortest path lengths from each top node
    nodes = list(G.nodes())
    dist_matrix = np.zeros((len(nodes), num_clusters))

    for i, seed in enumerate(top_nodes):
        shortest_paths = nx.single_source_shortest_path_length(G, seed)
        for j, node in enumerate(nodes):
            dist_matrix[j, i] = shortest_paths.get(node, np.inf)  # ∞ if no path

    dist_matrix[np.isinf(dist_matrix)] = np.finfo(np.float64).max
    # Apply KMeans clustering on influence distances
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(dist_matrix)

    # Assign communities based on clustering result
    node_communities = {nodes[i]: labels[i] for i in range(len(nodes))}

    # Print communities

     communities = defaultdict(list)
    for node, comm in node_communities.items():
        communities[comm].append(node)

    for comm, members in communities.items():
        print(f"Community {comm + 1}: {members}")
