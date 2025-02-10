import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict

def star_nodes(G):
    # Get degree for each node
    degree = dict(G.degree())
    star= {}
    nodes = list(G.nodes())
    for i in nodes:
        neighbors = G.neighbors(i)
        sum = 0
        for j in neighbors:
            sum += G.degree(j)
        star_value = sum/G.degree(i)
        star[i] = star_value

    # Prepare feature matrix using degree only (1D)
    degree_array = np.array([[star[n]] for n in nodes])

    # Cluster nodes using KMeans (to spread high-degree nodes)
    num_clusters = 30  # Define number of communities
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(degree_array)

    # Assign nodes to communities
    node_communities = {nodes[i]: labels[i] for i in range(len(nodes))}

    # Print communities
    communities = defaultdict(list)
    for node, comm in node_communities.items():
        communities[comm].append(node)

    for comm, members in communities.items():
        print(f"Community {comm + 1}: {len(members)}")
