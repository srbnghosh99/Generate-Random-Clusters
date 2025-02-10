import networkx as nx
from collections import defaultdict
from tqdm import tqdm


def assign_communities_based_on_threshold(graph, degree_threshold=None, attribute=None, attribute_threshold=None):
    # Dictionary to store the nodes in each community
    communities = defaultdict(list)

    # Iterate over each node in the graph
    for node in tqdm(graph.nodes(), desc="Assigning nodes to communities"):
        if degree_threshold is not None and graph.degree[node] > degree_threshold:
            communities['degree_based'].append(node)
        if attribute is not None and attribute_threshold is not None:
            if graph.nodes[node].get(attribute, 0) > attribute_threshold:
                communities['attribute_based'].append(node)

    return communities


if __name__ == "__main__":
    # Create an example graph
    G = nx.karate_club_graph()

    # Add node attributes for demonstration purposes
    for node in G.nodes():
        G.nodes[node]['weight'] = G.degree[node] + 1  # Example attribute

    # Set thresholds
    degree_threshold = 10
    attribute = 'weight'
    attribute_threshold = 15

    # Assign nodes to communities
    communities = assign_communities_based_on_threshold(G, degree_threshold, attribute, attribute_threshold)

    for community, nodes in communities.items():
        print(f"Community {community}: {nodes}")