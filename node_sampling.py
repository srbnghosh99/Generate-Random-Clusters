import networkx as nx
import random
from collections import defaultdict
import pandas as pd

def sample_nodes_to_communities(graph, num_communities, nodes_per_community):
    nodes = list(graph.nodes())
    random.shuffle(nodes)

    communities = defaultdict(list)

    for i, node in enumerate(nodes):
        community_id = i % num_communities
        communities[community_id].append(node)

    return communities


def random_node_samp(G):
    # G = nx.karate_club_graph()  # Example graph
    num_communities = 30
    nodes_per_community = len(G.nodes) // num_communities

    communitydict = sample_nodes_to_communities(G, num_communities, nodes_per_community)

    for community, nodes in communitydict.items():
        print(f"Community {community}: {len(nodes)}")

    community_df = pd.DataFrame(communitydict.items(), columns=['Community_id', 'Nodes'])
    community_df['len'] = community_df['Nodes'].apply(len)
    # community_df = community_df.reset_index()
    # community_df = community_df.rename(columns={'index': 'Community_id'})

    return community_df