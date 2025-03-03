import networkx as nx
import pandas as pd

def connected_components(G):
    commid = []
    nodeslist = []
    i = 0
    for c in sorted(nx.connected_components(G), key=len, reverse=True):
        print(len(c))
        commid.append(i)
        nodeslist.append(c)
        i += 1

    df = pd.DataFrame({
        'Community_id': commid,  # Keep Node_IDs for test set
        'Nodes': nodeslist,
    })
    df['len'] = df['Nodes'].apply(len)
    print(df)
    return df
