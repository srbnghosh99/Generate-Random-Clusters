import pandas as pd

def degree_based_comm(G):
    communitydict = G.degree()
    detected_community_df = pd.DataFrame.from_dict(communitydict)
    detected_community_df.columns = ['Node', 'Degree']
    detected_community_df.sort_values(by='Degree', ascending=False)
    communities = list(detected_community_df['Degree'].unique())
    communitydict = {}
    for i in communities:
        nodelist = detected_community_df[detected_community_df['Degree'] == i]['Node'].tolist()
        communitydict[i] = nodelist
    community_df = pd.DataFrame(communitydict.items(), columns=['Degree', 'Nodes'])
    community_df = community_df.sort_values(by='Degree', ascending=False)
    community_df['Community_id'] = community_df['Nodes'].apply(len)
    community_df = community_df.groupby('Community_id')['Nodes'].apply(list).reset_index()
    # print(community_df.keys)
    for index, row in community_df.iterrows():
        nodeslists = row['Nodes']
        flatList = [element for innerList in nodeslists for element in innerList]
        # print(flatList)
        community_df.at[index, 'Nodes'] = flatList
    community_df['len'] = community_df['Nodes'].apply(len)



    # for index, row in community_df.iterrows():
    #     i = row['Len']
    #     # print(i)
    #     nodes_to_include = community_df.loc[community_df['Len'] == i, 'Nodes'].iloc[0]
    #     # print(nodes_to_include)
    #     flatList = [element for innerList in nodes_to_include for element in innerList]
    #     row['Nodes'] = flatList
        # print(flatList)
        # subgraph = G.subgraph(flatList)
        # print(f'No of nodes: {subgraph.number_of_nodes()}, No of edges: {subgraph.number_of_edges()}')
    print(community_df.keys)
    # community_df.to_csv('random_graph_communities.csv', index = False)

