import networkx as nx
from collections import defaultdict

def update_dict_with_list(my_dict, key, values):
    # Ensure the key exists in the dictionary with a list
    if key not in my_dict:
        my_dict[key] = []

    # Append only values that are not already in the list
    my_dict[key].extend(v for v in values if v not in my_dict[key])
    # print(len(my_dict[key]))
    return my_dict

def finding_egos(G):
    communities = defaultdict(list)
    for node in G.nodes():
        S = nx.ego_graph(G, node, radius=2)
        len = S.number_of_nodes()

        communities = update_dict_with_list(communities,len,S.nodes())

    for comm, members in communities.items():
        # print(communities[comm])
        print(f"Community {comm + 1}: {(members)}")
