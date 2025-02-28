import networkx as nx
from collections import defaultdict
import pandas as pd

def update_dict_with_list(my_dict, key, values):
    # Ensure the key exists in the dictionary with a list
    if key not in my_dict:
        my_dict[key] = []

    # Append only values that are not already in the list
    my_dict[key].extend(v for v in values if v not in my_dict[key])
    # print(len(my_dict[key]))
    return my_dict

def finding_egos(G):
    communitydict = defaultdict(list)
    for node in G.nodes():
        S = nx.ego_graph(G, node, radius=1)
        len = S.number_of_nodes()

        communitydict = update_dict_with_list(communitydict,len,S.nodes())

    # for comm, members in communitydict.items():
    #     # print(communities[comm])
    #     print(f"Community {comm + 1}: {(members)}")
    # print(communitydict.keys)
    community_df = pd.DataFrame(communitydict.items(), columns=['Key', 'Nodes'])
    community_df['len'] = community_df['Nodes'].apply(len)
    community_df = community_df.reset_index()
    community_df = community_df.rename(columns={'index': 'Community_id'})
    community_df = community_df[['Community_id','len','Nodes']]
    print(community_df)
    return community_df


