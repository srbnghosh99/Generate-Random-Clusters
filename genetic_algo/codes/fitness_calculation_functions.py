from collections import defaultdict
import networkx as nx
import file_read_write
import pandas as pd
import numpy as np
import time
from itertools import combinations
# import plot_metrics
import argparse
import os
from tqdm import tqdm 


def create_nested_folder(path):
    current_path = ""
    for part in path.split(os.sep):
        if part == "":
            continue  # skip empty parts (for absolute paths)
        current_path = os.path.join(current_path, part)
        if not os.path.exists(current_path):
            os.mkdir(current_path)
            print(f"Created: {current_path}")
        else:
            print(f"Exists: {current_path}")

def create_folder(outdirectory):
    if os.path.exists(outdirectory):
        print('folder exists')
    else:
        try:
            print('create folder')
            os.mkdir(outdirectory)
            print(f"Directory '{outdirectory}' created successfully")
        except FileExistsError:
            print(f"Directory '{outdirectory}' already exists")
        except Exception as e:
            print(f"An error occurred: {e}")

def no_of_communities(master_populations):

    commlist = []
    for solution in master_populations:
        no_comm = len(set(solution))
        commlist.append(no_comm)
    return commlist


def fitness_calculation_modularity(master_populations,G):
    nodes = G.nodes()
    fitness_list = []

    for solution in tqdm(master_populations, desc="Processing list"):
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
    for solution in tqdm(master_populations, desc="Processing list"):
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
    for solution in tqdm(master_populations, desc="Processing list"):
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
    for solution in tqdm(master_populations, desc="Processing list"):
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
    for solution in tqdm(master_populations, desc="Processing list"):
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
    for solution in tqdm(master_populations, desc="Processing list"):
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


def parse_args():
    parser = argparse.ArgumentParser(description="Read File and Process Data")
    # Add a global argument for the input file
    parser.add_argument("--inputfile", type=str, required=True, help="Path to the input directory")
    # parser.add_argument("--metric1", type=str, required=True, help="Path to the input directory")
    parser.add_argument("--graphfile", type=str, required=True, help="Path to the input directory")
    parser.add_argument("--outfile", type=str, required=False, help="Path to the input directory")
    parser.add_argument("--opti_func", type=str, required=True, help="Path to the input directory")
    parser.add_argument("--start", type=int, required=False, help="Path to the input directory")
    parser.add_argument("--end", type=int, required=False, help="Path to the input directory")
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    G = nx.read_edgelist(args.graphfile)
    print('inputfile',args.inputfile)
    print('outputfile',args.outfile)
    create_nested_folder(args.outfile)
    solutions,fitness = file_read_write.load_solutions(args.inputfile)
    print(f'Number of solutions {len(solutions)}')
    if args.start is not None:
        start = args.start
        end = args.end
        solutions = solutions[start:end]
        fitness = fitness[start:end]


    if args.opti_func == 'modualrity':
        densitylis = fitness_calculation_density(solutions, G)
        no_of_communitieslis = no_of_communities(solutions)
        conductancelis = fitness_calculation_conductance(solutions, G)
        cutsizelis = fitness_calculation_cutsize(solutions, G)
        cclis = average_clustering_within_communities(solutions, G)
        centralitylis = community_centralization(solutions, G)
        df = pd.DataFrame({'Solution':solutions,'Fitness_val':fitness,'Avg_density':densitylis,'Communities':no_of_communitieslis,'Avg_Conductance':conductancelis,'Avg_Cut_size': cutsizelis,'Avg_Clustering_coeff':cclis,'avg_Centralization': centralitylis})
    elif args.opti_func == 'density':
        no_of_communitieslis = no_of_communities(solutions)
        conductancelis = fitness_calculation_conductance(solutions, G)
        cutsizelis = fitness_calculation_cutsize(solutions, G)
        cclis = average_clustering_within_communities(solutions, G)
        centralitylis = community_centralization(solutions, G)
        modularitylis = fitness_calculation_modularity(solutions, G)
        df = pd.DataFrame({'Solution': solutions, 'Fitness_val': fitness, 'Modularity': modularitylis,
                           'Communities': no_of_communitieslis, 'Avg_Conductance': conductancelis,
                           'Avg_Cut_size': cutsizelis, 'Avg_Clustering_coeff': cclis,
                           'avg_Centralization': centralitylis})
    elif args.opti_func == 'conductance':
        densitylis = fitness_calculation_density(solutions, G)
        no_of_communitieslis = no_of_communities(solutions)
        cutsizelis = fitness_calculation_cutsize(solutions, G)
        cclis = average_clustering_within_communities(solutions, G)
        centralitylis = community_centralization(solutions, G)
        modularitylis = fitness_calculation_modularity(solutions, G)
        df = pd.DataFrame({'Solution': solutions, 'Fitness_val': fitness, 'Modularity': modularitylis,
                           'Communities': no_of_communitieslis, 'Avg_density': densitylis,
                           'Avg_Cut_size': cutsizelis, 'Avg_Clustering_coeff': cclis,
                           'avg_Centralization': centralitylis})
    elif args.opti_func == 'cc':
        densitylis = fitness_calculation_density(solutions, G)
        no_of_communitieslis = no_of_communities(solutions)
        conductancelis = fitness_calculation_conductance(solutions, G)
        cutsizelis = fitness_calculation_cutsize(solutions, G)
        centralitylis = community_centralization(solutions, G)
        modularitylis = fitness_calculation_modularity(solutions, G)
        df = pd.DataFrame({'Solution': solutions, 'Fitness_val': fitness, 'Modularity': modularitylis,
                           'Communities': no_of_communitieslis, 'Avg_density': densitylis,
                           'Avg_Cut_size': cutsizelis, 'Avg_Conductance': conductancelis,
                           'avg_Centralization': centralitylis})

    # df = df.sort_values(by=['Avg_density'],ascending = False)


    print(df)
    print('done')
    #create_folder(args.outfile)
    
    outputfile = args.outfile + '/' + str(args.start) + 'to' + str(args.end) + '.csv'
    df.to_csv(outputfile)
'''
df = pd.read_csv('output_ciao/output3_metrics.csv')
columns = ['Avg_density', 'Avg_Conductance','Avg_Cut_size', 'Avg_Clustering_coeff', 'avg_Centralization']
count = 0
for metric1, metric2 in combinations(columns, 2):
    filename = 'figure_' + str(count) + '.png'
    print(filename)
    plot_metrics.plot_scatter_func(df, metric1, metric2, filename)
    count += 1


# plot_metrics.plot_scatter(df,'Avg_density','Avg_Conductance')
# plot_metrics.plot_scatter(df,'Avg_density','Avg_Conductance')

end_time = time.time()
elapsed_time = end_time - start_time
elapsed_time = elapsed_time/60
print(f"Elapsed time: {elapsed_time:.6f} minutes")
    # df = pd.DataFrame({'Solution':solutions,'Fitness_value':conductancelis })
    # df = pd.DataFrame({'Solution':solutions,'Fitness_value':cutsizelis })
    # df = pd.DataFrame({'Solution':solutions,'Fitness_value':cclis })
    # df = pd.DataFrame({'Solution':solutions,'Avg_density':densitylis,'no_of_communities':no_of_communitieslis  })
    # df = pd.DataFrame({'Solution':solutions,'Modularity':centralitylis })
    # df = df.sort_values(by=['Fitness_value'],ascending = False)
    # df = pd.DataFrame({'Solution':solutions,'no_of_communities':no_of_communitieslis })
        # df = df.sort_values(by=['no_of_communities'],ascending = False)
        
    # print('density calc')
    # densitylis = fitness_calculation_density(solutions,G)
    # no_of_communitieslis =  no_of_communities(solutions)
    # print('conductance calc')
    # conductancelis = fitness_calculation_conductance(solutions,G)
    # print('cutsize calc')
    # cutsizelis = fitness_calculation_cutsize(solutions,G)
    # print('cluster coeff calc')
    # cclis = average_clustering_within_communities(solutions,G)
    # print('density centralization')
    # centralitylis = community_centralization(solutions,G)
    # print('completed calculation')
    # modularitylis = fitness_calculation_modularity(solutions, G)
    # print('completed calculation')
    
    # solutions, fitness_values1 = file_read_write.load_solutions(args.inputfile)
    # filename = 'otherfiles/ciao.bson'
    # # filename = 'output_ciao/output3.bson'
    # G = nx.read_edgelist("../randomComm/graph_files/renumbered_graph_ciao.csv")

    #solutions = file_read_write.load_solutions_v1(args.inputfile)
'''
