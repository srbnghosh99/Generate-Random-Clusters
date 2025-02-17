import networkx as nx
from collections import defaultdict


def finding_cliques(G):
    # Compute cliques count
    cliques = list(nx.find_cliques(G))  # List of all maximal cliques
    num_cliques = len(cliques)
    communities = defaultdict(list)
    for clique in cliques:
        len_clique = len(clique)
        communities[len_clique].append(clique)
    for comm, members in communities.items():
        print(f"Community {comm + 1}: {len(members)}")

    # return communities
    

