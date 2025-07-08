from collections import defaultdict
import networkx as nx
import file_read_write
import pandas as pd
import numpy as np
import time
from itertools import combinations
# import plot_metrics
import argparse


def fitness_calculation_modularity(master_populations,G):
    nodes = G.nodes()
    fitness_list = []

    for solution in master_populations:
        community_dict = defaultdict(list)
        community_ids = solution

        for node, community in zip(nodes, community_ids):
            community_dict[community].append(node)
        # Convert to a list of lists
        grouped_nodes = list(community_dict.values())
        mod = nx.community.modularity(G, grouped_nodes)
        fitness_list.append(mod)
    return fitness_list

def fitness_calculation_density(master_populations,G):
    # print("fitness_calculation_density")
    nodes = G.nodes()
    fitness_list = []
    for solution in master_populations:
        # community_dict = defaultdict(list)
        # community_ids = solution
        community_groups = defaultdict(list)
        community_ids = solution
        for node, community in zip(nodes, community_ids):
            community_groups[community].append(node)
        df = pd.DataFrame(community_groups.items(), columns=['Community_id', 'Nodes'])
        df['len'] = df['Nodes'].apply(len)
        df = df.sort_values(by=['len'], ascending=False)
        density = []
        total_weight = 0
        weighted_density_sum = 0
        for index, row in df.iterrows():
            size = len(row['Nodes'])
            # if len(row['Nodes']) >= 10:
            subgraph = G.subgraph(row['Nodes'])
            weight = size  # or use weight = size * (size - 1) / 2 for number of node pairs
            density = nx.density(subgraph)
            weighted_density_sum += density * weight
            total_weight += weight

        val = weighted_density_sum / total_weight if total_weight > 0 else 0
        fitness_list.append(val)

    return fitness_list

def fitness_calculation_cutsize(master_populations,G):
    # print("fitness_calculation_cutsize")
    nodes = G.nodes()

    fitness_list = []
    for solution in master_populations:
        community_groups = defaultdict(list)
        community_ids = solution
        for node, community in zip(nodes, community_ids):
            community_groups[community].append(node)
        df = pd.DataFrame(community_groups.items(), columns=['Community_id', 'Nodes'])
        df['len'] = df['Nodes'].apply(len)
        df = df.sort_values(by=['len'], ascending=False)

        communities = sorted(community_groups.keys())
        # print(f'Number of communities {len(communities)}')
        # if len(communities) >= G.number_of_nodes():
        #     fitness_list.append(0.0)
        #     continue
        cut_matrix = pd.DataFrame(0.0, index=communities, columns=communities)

        # Compute cut size between each community pair
        for c1, c2 in combinations(communities, 2):
            cut = nx.cut_size(G, community_groups[c1], community_groups[c2])
            cut_matrix.loc[c1, c2] = cut
            cut_matrix.loc[c2, c1] = cut
        avg_cut_matrix = np.mean(cut_matrix)
        # print(avg_cut_matrix)
        fitness_list.append(avg_cut_matrix)
    return fitness_list

def fitness_calculation_conductance(master_populations, G):
    # print("fitness_calculation_conductance")
    # print(G.number_of_nodes())
    nodes = G.nodes()
    fitness_list = []
    for solution in master_populations:
        # community_dict = defaultdict(list)
        # community_ids = solution
        community_groups = defaultdict(list)
        community_ids = solution
        # print(len(community_ids),len(nodes))
        for node, community in zip(nodes, community_ids):
            community_groups[community].append(node)
        df = pd.DataFrame(community_groups.items(), columns=['Community_id', 'Nodes'])
        df['len'] = df['Nodes'].apply(len)
        df = df.sort_values(by=['len'], ascending=False)

        communities = sorted(community_groups.keys())
        # print(f'Number of communities {len(communities)}')
        # if len(communities) > 1000:
        #     fitness_list.append(0.0)
        #     continue
        conductance = pd.DataFrame(0.0, index=communities, columns=communities)
        # Compute cut size between each community pair
        for c1, c2 in combinations(communities, 2):
            conduc = nx.conductance(G, community_groups[c1], community_groups[c2])
            # print(cut)
            conductance.loc[c1, c2] = conduc
            conductance.loc[c2, c1] = conduc
        # print(c1,c2,conduc)

        avg_conductance = np.mean(conductance)
        # print(avg_conductance)
        fitness_list.append(avg_conductance)
    return fitness_list

def average_clustering_within_communities(master_populations,G ):
    # print("average_clustering_within_communities")
    nodes = G.nodes()
    fitness_list = []
    for solution in master_populations:
        community_groups = defaultdict(list)
        community_ids = solution
        for node, community in zip(nodes, community_ids):
            community_groups[community].append(node)

        df = pd.DataFrame(community_groups.items(), columns=['Community_id', 'Nodes'])
        df['len'] = df['Nodes'].apply(len)
        df = df.sort_values(by=['len'], ascending=False)
        # print(f'Number of communities {len(communities)}')
        # if len(communities) > 1000:
        #     fitness_list.append(0.0)
        #     continue
        total_cc = 0
        for index, row in df.iterrows():
            subgraph = G.subgraph(row['Nodes'])
            cc = nx.average_clustering(subgraph)
            total_cc += cc * len(subgraph)  # weighted sum

        avg_cc = total_cc / G.number_of_nodes()
        # fitness_list.append(np.mean(avg_cc))
        fitness_list.append(avg_cc)
    return fitness_list

def community_centralization(master_populations,G):
    nodes = G.nodes()
    fitness_list = []
    for solution in master_populations:
        community_groups = defaultdict(list)
        community_ids = solution
        for node, community in zip(nodes, community_ids):
            community_groups[community].append(node)
        df = pd.DataFrame(community_groups.items(), columns=['Community_id', 'Nodes'])
        df['len'] = df['Nodes'].apply(len)
        df = df.sort_values(by=['len'], ascending=False)
        # print(f'number of communities {df.shape[0]}')
        scores = []
        for index, row in df.iterrows():
            # community = row['Community_id']
            # print(f'number of communities {community}')
            total = 0
            subgraph = G.subgraph(row['Nodes'])
            centrality = nx.degree_centrality(subgraph)  # or closeness/betweenness
            max_c = max(centrality.values())
            total = sum(max_c - c for c in centrality.values())
            scores.append(total)
        fitness_list.append(np.mean(scores))
    return fitness_list