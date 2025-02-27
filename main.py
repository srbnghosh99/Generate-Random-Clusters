# This is a sample Python script.
import pandas as pd
import networkx as nx
import numpy as np
import argparse
from sklearn.cluster import KMeans
from collections import defaultdict
import igraph as ig
import random
import node_sim
import node_sampling
import random_walk
import star_nodes
import connected_component
import influence_spread
import degree_based
import graph_traversal
import clique_clusters
import ego_clusters


    #     commid = commid + 1
    # detected_community_df = pd.DataFrame.from_dict(communitydict, orient='index').reset_index()
    # detected_community_df.columns = ['Node', 'Community']
    # community_df = detected_community_df.groupby('Community')['Node'].apply(list).reset_index()
    # print(community_df['Community'].nunique())


def parse_args():
    parser = argparse.ArgumentParser(description="Read File and Process Data")
    # Add a global argument for the input file
    parser.add_argument("--inputfilename", type=str, required=True, help="Path to the input file")

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help="Available commands")

    # Subparser for the 'degree_based_comm' command
    parser_degree = subparsers.add_parser('degree_based_comm', help="Perform degree-based community detection")
    parser_cc = subparsers.add_parser('connected_components', help="Perform connected components community detection")
    parser_influence = subparsers.add_parser('influence_spread', help="Perform connected components community detection")
    parser_star = subparsers.add_parser('star_nodes',
                                             help="Perform connected components community detection")
    parser_random_walk = subparsers.add_parser('random_walk',
                                        help="Perform connected components community detection")
    parser_node_sim = subparsers.add_parser('node_similarity',
                                               help="Perform connected components community detection")
    parser_node_sampling = subparsers.add_parser('random_node_sampling',
                                               help="Perform connected components community detection")
    parser_graph_travers = subparsers.add_parser('graph_traversal',
                                               help="Perform connected components community detection")
    parser_clique_clusters = subparsers.add_parser('clique_clusters',
                                                 help="Perform connected components community detection")
    parser_ego_clusters = subparsers.add_parser('ego_clusters',
                                                 help="Perform connected components community detection")

    parser_degree.add_argument('--outputfilename', type=str, required=True, help="Path to the output file")
    parser_cc.add_argument('--outputfilename', type=str, required=True, help="Path to the output file")
    parser_influence.add_argument('--outputfilename', type=str, required=True, help="Path to the output file")
    parser_star.add_argument('--outputfilename', type=str, required=True, help="Path to the output file")
    parser_random_walk.add_argument('--outputfilename', type=str, required=True, help="Path to the output file")
    parser_node_sim.add_argument('--outputfilename', type=str, required=True, help="Path to the output file")
    parser_node_sampling.add_argument('--outputfilename', type=str, required=True, help="Path to the output file")
    parser_graph_travers.add_argument('--outputfilename', type=str, required=True, help="Path to the output file")
    parser_clique_clusters.add_argument('--outputfilename', type=str, required=True, help="Path to the output file")
    parser_ego_clusters.add_argument('--outputfilename', type=str, required=True, help="Path to the output file")
    # parser_node_sim.add_argument('--outputfilename', type=str, required=True, help="Path to the output file")
    return parser.parse_args()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    args = parse_args()
    print(args.inputfilename)
    # print(args.feature)

    # if args.command == 'neighbors':
    #     initial(args.inputfilename,)
    Graph = nx.read_edgelist(args.inputfilename)
    print(Graph.number_of_nodes(), Graph.number_of_edges())

    if args.command == 'degree_based_comm':
        print(f"Running degree-based community detection on {args.inputfilename}")
        print(f"Output will be saved to {args.outputfilename}.")
        degree_based.degree_based_comm(Graph)
    elif args.command == 'connected_components':
        print(f"Running connected component based community detection on {args.inputfilename}")
        print(f"Output will be saved to {args.outputfilename}.")
        connected_component.connected_components(Graph)
    elif args.command == 'influence_spread':
        print(f"Running influence spread based community detection on {args.inputfilename}")
        print(f"Output will be saved to {args.outputfilename}.")
        df = influence_spread.influence_spread_communities(Graph)
        # print(df)
        df.to_csv(args.outputfilename,index = False)
    elif args.command == 'star_nodes':
        print(f"Running star nodes based community detection on {args.inputfilename}")
        print(f"Output will be saved to {args.outputfilename}.")
        df = star_nodes.star_nodes(Graph)
        df.to_csv(args.outputfilename, index=False)
    elif args.command == 'random_walk':
        print(f"Running star nodes based community detection on {args.inputfilename}")
        print(f"Output will be saved to {args.outputfilename}.")
        random_walk.random_walk2(Graph)
    elif args.command == 'node_similarity':
        print(f"Running star nodes based community detection on {args.inputfilename}")
        # print(f"Output will be saved to {args.outputfilename}.")
        node_sim.detect_communities(Graph)
    elif args.command == 'random_node_sampling':
        print(f"Running star nodes based community detection on {args.inputfilename}")
        # print(f"Output will be saved to {args.outputfilename}.")
        node_sampling.random_node_samp(Graph)
    elif args.command == 'graph_traversal':
        print(f"Running star nodes based community detection on {args.inputfilename}")
        # print(f"Output will be saved to {args.outputfilename}.")
        graph_traversal.bfs_communities(Graph)
    elif args.command == 'clique_clusters':
        print(f"Running star nodes based community detection on {args.inputfilename}")
        # print(f"Output will be saved to {args.outputfilename}.")
        clique_clusters.finding_cliques(Graph)
    elif args.command == 'ego_clusters':
        print(f"Running ego nets based community detection on {args.inputfilename}")
        # print(f"Output will be saved to {args.outputfilename}.")
        ego_clusters.finding_egos(Graph)
        df = ego_clusters.finding_egos(Graph)
        df.to_csv(args.outputfilename, index=False)

    else:
        print("No command selected. Use --help for usage information.")
