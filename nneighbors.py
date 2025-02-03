# This is a sample Python script.
import pandas as pd
import networkx as nx
import numpy as np
import argparse

def find_neighbors(node,communitydict,commid):
    neighborslist = list(G.neighbors(node))
    for j in neighborslist:
        if j in communitydict:
            continue
        else:
            communitydict[j] = commid

def parse_args():
    parser = argparse.ArgumentParser(description="Read File")
    parser.add_argument("--inputfilename",type = str)
    parser.add_argument("--outputfilename",type = str)
    return parser.parse_args()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # inputs = parse_args()
    # print(inputs.inputfilename)
    # print(inputs.outputfilename)
    G = nx.read_edgelist("/Users/shrabanighosh/UNCC/Spring 2025/Random communities generation/trustnetwork.csv")
    G.number_of_nodes(), G.number_of_edges()
    communitydict = {}
    commid = 1
    for node in G.nodes():
        if node not in communitydict:
            communitydict[node] = commid
        find_neighbors(node, communitydict, commid)
        commid = commid + 1
    detected_community_df = pd.DataFrame.from_dict(communitydict, orient='index').reset_index()
    detected_community_df.columns = ['Node', 'Community']
    community_df = detected_community_df.groupby('Community')['Node'].apply(list).reset_index()
    print(community_df['Community'].nunique())