from math import floor
from symbol import return_stmt
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ast
import random
import preprocess
import os
import glob
from collections import defaultdict
import argparse
import sys
import pickle
from community import community_louvain
from tqdm import tqdm
import time
from collections import Counter
import shutil
import selection_solution_process
import file_read_write
# import fitness_calculation_functions
from functools import wraps
import optimization_functions


master_populations = []
iteration = 0
best_modularity_per_gen = []
modularity_per_gen_avg = []
modularity_per_gen_median = []

# def timer(func):
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         start_time = time.time()
#         result = func(*args, **kwargs)
#         end_time = time.time()
#         print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds")
#         return result
#     return wrapper

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
        for filename in os.listdir(outdirectory):
            file_path = os.path.join(outdirectory, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        try:
            print('create folder')
            os.mkdir(outdirectory)
            print(f"Directory '{outdirectory}' created successfully")
        except FileExistsError:
            print(f"Directory '{outdirectory}' already exists")
        except Exception as e:
            print(f"An error occurred: {e}")

def random_init(G,nocommunities):
    num_of_comm = list(range(0,nocommunities))

    comm_lis = []
    # for i in range(0,20):
    #     comm_lis[i] = []
    nodelis =  list(G.nodes())
    # nodelis
    # print(nodelis)
    for i in nodelis:

        random_comm = random.choice(num_of_comm)
        comm_lis.append(random_comm)
        # comm_lis[random_comm].append(i)
    # print(comm_lis)
    return comm_lis


def process_population(master_populations, G, fitness_list, crossover_nodes, mutation_nodes, outfile,popusize,max_iterations,function):
    global iteration
    start_time = time.time()
    popusize = int(popusize)
    max_iterations = int(max_iterations)

    with tqdm(total=max_iterations, desc="Genetic Algorithm Progress", dynamic_ncols=True) as pbar:
        while max(fitness_list) > 0.0:
            iteration += 1
            if iteration == max_iterations:
                print("Reached maximum iteration, no solution found.")
                break
            offspringsolutions, parent_fitness = community_based_crossover(G, fitness_list, crossover_nodes, outfile)
            master_populations, fitness_list = neighborhood_based_mutation(offspringsolutions, G, function, mutation_rate=0.1,)
            # offspringsolutions, parent_fitness = crossover(G, fitness_list, crossover_nodes,outfile)
            # master_populations, fitness_list = mutation(G, offspringsolutions, fitness_list, parent_fitness, mutation_nodes,outfile)
            best_modularity_per_gen.append(max(fitness_list))
            modularity_per_gen_avg.append(np.mean(fitness_list))
            modularity_per_gen_median.append(np.median(fitness_list))
            # print()
            pbar.update(1)
        print("Reached maximum solutions.")
        print('Number of iteration', iteration)
        print("Number of solutions", len(master_populations))
        end_time = time.time()
        duration = end_time - start_time
        duration = duration / 60
        print(f"The Processing population took {duration:.2f} minutes to run.")
        file_read_write.write_to_file(outfile, f'Number of solutions generated, {len(master_populations)}')
        file_read_write.write_to_file(outfile,f'The Processing population took {duration:.2f} minutes to run.')
        return master_populations,fitness_list
    file_read_write.write_to_file(outfile, f'Number of solutions generated, {len(master_populations)}')
    file_read_write.write_to_file(outfile, f'The Processing population took {duration:.2f} minutes to run.')
    return master_populations,fitness_list

'''
def fitness_calculation(master_populations,G):
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
'''
def normalize_communities(community_list):
    mapping = {}
    next_id = 0
    normalized = []
    for c in community_list:
        if c not in mapping:
            mapping[c] = next_id
            next_id += 1
        normalized.append(mapping[c])
    return normalized

def community_based_crossover(G, fitness_list, crossover_nodes,outfile):
    sampled_indices = random.sample(range(len(fitness_list)), 4)
    # print(fitness_list)
    # Retrieve corresponding (index, fitness) pairs
    indexed_fitness = [(i, fitness_list[i]) for i in sampled_indices]

    # Sort the sampled solutions based on fitness value (descending)
    indexed_fitness.sort(key=lambda x: x[1], reverse=True)

    # Pick two parents with the highest fitness values
    index_of_parent1, parent1_fitness = indexed_fitness[0]
    index_of_parent2, parent2_fitness = indexed_fitness[1]
    # Retrieve parent solutions from master_populations
    parent1 = master_populations[index_of_parent1][:]
    parent2 = master_populations[index_of_parent2][:]
    # print(parent1)

    child = [-1] * len(parent1)  # Assuming list of community assignments

    # Step 1: Pick a random community from parent1
    chosen_community = random.choice(list(set(parent1)))

    # Step 2: Copy nodes from that community
    for idx, community in enumerate(parent1):
        if community == chosen_community:
            child[idx] = community

    # Step 3: Fill remaining nodes from parent2
    for idx, val in enumerate(child):
        if val == -1:
            child[idx] = parent2[idx]
    offspring = normalize_communities(child)
    return offspring, [parent1_fitness,parent2_fitness]

def neighborhood_based_mutation(individual, G,function, mutation_rate=0.1):
    mutated = individual.copy()
    node_list = list(G.nodes())
    node_to_index = {node: i for i, node in enumerate(node_list)}

    for node in node_list:
        if random.random() < mutation_rate:
            neighbors = list(G.neighbors(node))
            if not neighbors:
                continue

            # Collect neighbors' communities using mapping
            neighbor_communities = []
            for nbr in neighbors:
                if nbr in node_to_index:
                    neighbor_communities.append(individual[node_to_index[nbr]])

            if neighbor_communities:
                # Choose the most common neighbor community
                most_common = Counter(neighbor_communities).most_common(1)
                best_community = most_common[0][0]

                # Mutate current node to this community
                mutated[node_to_index[node]] = best_community

    # offsprings_fitness = fitness_calculation([mutated], G)
    if function == 'modularity':
        offsprings_fitness = optimization_functions.fitness_calculation_modularity([mutated], G)
    if function == 'density':
        offsprings_fitness = optimization_functions.fitness_calculation_density([mutated], G)
    if function == 'cutsize':
        offsprings_fitness = optimization_functions.fitness_calculation_cutsize([mutated], G)
    if function == 'conductance':
        offsprings_fitness = optimization_functions.fitness_calculation_conductance([mutated], G)
    if function == 'clustering_coeff':
        offsprings_fitness = optimization_functions.average_clustering_within_communities([mutated], G)
    fitness_list.append(offsprings_fitness[0])
    master_populations.append(mutated)
    return master_populations, fitness_list


def parse_args():
    parser = argparse.ArgumentParser(description="Read File and Process Data")
    # Add a global argument for the input file
    parser.add_argument("--inputdir", type=str, required=True, help="Path to the input directory")
    parser.add_argument("--flag", type=str, required=True, help="Path to the input directory")
    parser.add_argument("--graphfile", type=str, required=True, help="Path to the input graph")
    parser.add_argument("--crossover_nodes", type=str, required=True, help="Number of nodes in the crossover stage")
    parser.add_argument("--mutation_nodes", type=str, required=True, help="Number of nodes in the mutation stage")
    parser.add_argument("--outputfile", type=str, required=True, help="Output file name")
    parser.add_argument("--populationsize", type=str, required=True, help="Output file name")
    parser.add_argument("--iteration", type=str, required=True, help="Output file name")
    parser.add_argument("--function", type=str, required=True, help="Output file name")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print('inputfile: ',args.inputdir)
    print(args.flag)
    print(args.graphfile)
    print(args.crossover_nodes)
    print(args.mutation_nodes)
    print('outputfile: ',args.outputfile)
    print(args.populationsize)
    print(args.iteration)
    start_time = time.time()
    directory = args.inputdir
    # print(Graph.number_of_nodes())
    G = nx.read_edgelist(args.graphfile)
    if args.flag == '1':
        files = glob.glob(directory + "/*.csv")
        for file in files:
            print(file)
            solution = preprocess.process_solutions(file,G)
            master_populations.append(solution)
    if args.flag == '3':
        nocomms = 140
        for i in range(100):
            comm_population = random_init(G,nocomms)
            master_populations.append(comm_population)
    if args.flag == '2':
        files = glob.glob(directory + "/*.bson")
        master_populations = []
        master_fitness = []
        for file in files:
            solutions, fitness_values = file_read_write.load_solutions(file)
            master_populations.extend(solutions)
            master_fitness.extend(fitness_values)
            print(len(master_populations),len(master_fitness))
        if len(master_populations) > 100000:
            random_indices = random.sample(range(len(master_populations)), 100000)
            master_populations = [master_populations[i] for i in random_indices]
            master_fitness = [master_fitness[i] for i in random_indices]

    print("Number of initial populations",len(master_populations))
    if os.path.exists(args.outputfile):
        os.remove(args.outputfile)
    if args.function == 'modularity':
        fitness_list = optimization_functions.fitness_calculation_modularity(master_populations, G)
    if args.function == 'density':
        fitness_list = optimization_functions.fitness_calculation_density(master_populations,G)
    if args.function == 'cutsize':
        fitness_list = optimization_functions.fitness_calculation_cutsize(master_populations, G)
    if args.function == 'conductance':
        fitness_list = optimization_functions.fitness_calculation_conductance(master_populations, G)
    if args.function == 'clustering_coeff':
        fitness_list = optimization_functions.average_clustering_within_communities(master_populations, G)
    # print('fitness_list',(fitness_list))
    file_read_write.write_to_file(args.outputfile,f'standard method label propagation modularity value, {nx.community.modularity(G, nx.community.label_propagation_communities(G))}')
    # write_to_file(args.outputfile,f'standard method louvain modularity value,{nx.community.modularity(G, nx.community.louvain_communities(G, seed=123))}')
    file_read_write.write_to_file(args.outputfile,f'Number of initial populations, {len(master_populations)}')
    file_read_write.write_to_file(args.outputfile,f'Number of iterations given as argument, {args.iteration}')
    master_populations, fitness_list = process_population(master_populations,G,fitness_list,args.crossover_nodes,args.mutation_nodes, args.outputfile,args.populationsize,args.iteration,args.function)
    combined = selection_solution_process.select_solutions(master_populations, fitness_list, G, args.outputfile)
    # combined = select_solutions(master_populations, fitness_list, G, args.outputfile)
    outdir = os.path.splitext(args.outputfile)[0]
    
    file = outdir
      
    file_read_write.save_bson_chunks(combined,outdir,args.function,args.outputfile)
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time = elapsed_time / 60
    print(f"Total Elapsed time: {elapsed_time:.6f} minutes")
    file_read_write.write_to_file(args.outputfile,f'Total Elapsed time: {elapsed_time:.6f} minutes')
    
    plt.plot(best_modularity_per_gen)
    plt.xlabel("Generation")
    plt.ylabel(f'Best {args.function}')
    plt.title("Fitness Improvement Over Generations")
    plt.grid(True)
    plt.savefig(outdir +'/best.png')
    # plt.show()
    plt.close()

    plt.plot(modularity_per_gen_median)
    plt.xlabel("Generation")
    plt.ylabel(f'Median {args.function}')
    plt.title("Fitness Improvement Over Generations")
    plt.grid(True)
    plt.savefig(outdir +'/med.png')
    # plt.show()
    plt.close()

    plt.plot(modularity_per_gen_avg)
    plt.xlabel("Generation")
    plt.ylabel(f'Average {args.function}')
    plt.title("Fitness Improvement Over Generations")
    plt.grid(True)
    plt.savefig(outdir +'/avg.png')
    # plt.show()
    plt.close()

