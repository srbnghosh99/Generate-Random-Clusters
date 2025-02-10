def random_walk(graph, start_node, walk_length):
    walk = [start_node]
    for _ in range(walk_length - 1):
        neighbors = list(graph.neighbors(walk[-1]))
        if not neighbors:
            break
        next_node = random.choice(neighbors)
        walk.append(next_node)
    return walk

def build_transition_matrix(graph):
    n = graph.number_of_nodes()
    transition_matrix = np.zeros((n, n))
    for i, node in enumerate(graph.nodes()):
        neighbors = list(graph.neighbors(node))
        if neighbors:
            prob = 1 / len(neighbors)
            for neighbor in neighbors:
                transition_matrix[i, list(graph.nodes()).index(neighbor)] = prob
    return transition_matrix

def cluster_graph(graph, num_clusters, walk_length=10, num_walks=100):
    transition_matrix = build_transition_matrix(graph)
    n = graph.number_of_nodes()
    walks = []
    for node in graph.nodes():
        for _ in range(num_walks):
            walk = random_walk(graph, node, walk_length)
            walks.append(walk)

    walk_count_matrix = np.zeros((n, n))
    for walk in walks:
        for i in range(len(walk)):
            for j in range(i+1, len(walk)):
                node_i = list(graph.nodes()).index(walk[i])
                node_j = list(graph.nodes()).index(walk[j])
                walk_count_matrix[node_i, node_j] += 1
                walk_count_matrix[node_j, node_i] += 1


    kmeans = KMeans(n_clusters=num_clusters)
    clusters = kmeans.fit_predict(walk_count_matrix)
    return clusters

def random_walk2(G):
    # G = nx.karate_club_graph()  # Example graph
    num_clusters = 4
    clusters = cluster_graph(G, num_clusters)
    print('clusters',clusters)