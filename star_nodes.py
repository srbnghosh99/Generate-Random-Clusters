import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
import pandas as pd
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
        star_value = G.degree(i)/sum
        star[i] = round(star_value, 4)
        if star_value == 1:
            print(i)

    # print(star)
    # Group by values
    grouped = defaultdict(list)
    for key, value in star.items():
        grouped[value].append(key)

    # Convert to a standard dictionary
    communitydict = dict(grouped)
    community_df = pd.DataFrame(communitydict.items(), columns=['star_value', 'Nodes'])
    community_df['len'] = community_df['Nodes'].apply(len)
    community_df = community_df.reset_index()
    community_df = community_df.rename(columns={'index': 'Community_id'})
    # community_df = community_df.reset_index('Community_id')
    print(community_df)
    return community_df
    # Prepare feature matrix using degree only (1D)
    # degree_array = np.array([[star[n]] for n in nodes])
    #
    # # Cluster nodes using KMeans (to spread high-degree nodes)
    # num_clusters = 30  # Define number of communities
    # kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    # labels = kmeans.fit_predict(degree_array)
    #
    # # Assign nodes to communities
    # node_communities = {nodes[i]: labels[i] for i in range(len(nodes))}

    # Print communities
    # communities = defaultdict(list)
    # for node, comm in node_communities.items():
    #     communities[comm].append(node)
    #
    # for comm, members in communities.items():
    #     print(f"Community {comm + 1}: {len(members)}")

