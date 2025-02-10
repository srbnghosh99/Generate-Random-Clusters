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
    community_df['Len'] = community_df['Nodes'].apply(len)
    community_df = community_df.groupby('Len')['Nodes'].apply(list).reset_index()
    for index, row in community_df.iterrows():
        i = row['Len']
        # print(i)
        nodes_to_include = community_df.loc[community_df['Len'] == i, 'Nodes'].iloc[0]
        # print(nodes_to_include)
        flatList = [element for innerList in nodes_to_include for element in innerList]
        # print(flatList)
        subgraph = G.subgraph(flatList)
        print(f'No of nodes: {subgraph.number_of_nodes()}, No of edges: {subgraph.number_of_edges()}')