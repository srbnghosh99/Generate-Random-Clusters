import networkx as nx
from collections import deque, defaultdict
from tqdm import tqdm
import pandas as pd
def bfs_communities(graph):
    visited = set()
    communities = []

    # Iterate through all nodes in the graph
    for start_node in tqdm(graph.nodes(), desc="Exploring communities with BFS"):
        if start_node not in visited:
            # Perform BFS from this start node
            community = []
            queue = deque([start_node])
            visited.add(start_node)

            while queue:
                node = queue.popleft()
                community.append(node)

                # Explore neighbors
                for neighbor in graph.neighbors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            communities.append(community)

    communitydict = defaultdict(list)
    nodelist = graph.nodes()
    # print(nodelist)
    communities = communities[0]
    print(communities)
    for node, community in zip(nodelist, communities):
        community = int(community)
        communitydict[community].append(node)

    # for index in range(0,len(communities)):
    #     print(index)
    #     id = communities[index]
    #     print('id', id)
    #     node = nodelist[0]
    #
    #     print('node',node)
    #     # communitydict[id].append(node)

    community_df = pd.DataFrame(communitydict.items(), columns=['Community_ID', 'Nodes'])
    # community_df.sorted
    community_df=community_df.sort_values(by=['Community_ID'])
    # df = pd.DataFrame({'Node': range(len(communities)), 'Community_ID': communities})
    print(community_df)
    return community_df