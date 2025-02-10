import networkx as nx
from collections import deque, defaultdict
from tqdm import tqdm

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

    return communities