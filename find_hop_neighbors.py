import networkx as nx

def find_nearestneighbors(G,node,communitydict,commid):
    neighborslist = list(G.neighbors(node))
    for j in neighborslist:
        if j in communitydict:
            continue
        else:
            communitydict[j] = commid

def find_neighbors_hop_distance(G, node,communitydict,commid,hop):
    # k = 1
    neighborslist = list(G.neighbors(node))
    for j in neighborslist:
        if j in communitydict:
            continue
        else:
            communitydict[j] = commid
    if hop == 3:
        return communitydict
    for j in neighborslist:
        # k += 1
        communitydict = find_neighbors_hop_distance(j,communitydict,commid,hop)
    return communitydict

def initial(inputfilename):
    G = nx.read_edgelist(inputfilename)
    print(G.number_of_nodes(), G.number_of_edges())
    communitydict = {}
    commid = 1
    hop = 3
    for node in G.nodes():
        if node not in communitydict:
            communitydict[node] = commid
        # find_nearestneighbors(G, node, communitydict, commid)
        communitydict = find_neighbors_hop_distance(G, node, communitydict, commid, hop)