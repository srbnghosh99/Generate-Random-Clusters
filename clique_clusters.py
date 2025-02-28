import networkx as nx
from collections import defaultdict
import pandas as pd


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
    community_df = pd.DataFrame(communities.items(), columns=['clique_value', 'Nodes'])

    for index, row in community_df.iterrows():
        nodeslists = row['Nodes']
        flatList = [element for innerList in nodeslists for element in innerList]
        # print(flatList)
        list(set(flatList))
        community_df.at[index, 'Nodes'] = list(set(flatList))


    community_df['len'] = community_df['Nodes'].apply(len)

    community_df = community_df.reset_index()
    community_df = community_df.rename(columns={'index': 'Community_id'})
    print(community_df)
    return community_df

