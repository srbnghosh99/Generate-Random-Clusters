from math import floor

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
import bson
from tqdm import tqdm

master_populations = []

iteration = 0
stoppping_criteria = 10000

# max_iterations = 8  # Set a max iteration limit
# iteration = 0  # Global variable to track iterations




# def prunning(master_populations):
def select_solutions(master_populations,G, retain_ratio=0.2, prob_ratio=0.6, diverse_count=10):
    # fitness_list = []
    # for solution in master_populations:
    fitness_list = (fitness_calculation(master_populations,G))
    df = pd.DataFrame({'Solution':master_populations,'Fitness_value':fitness_list})
    df = df.sort_values(by = ['Fitness_value'],ascending = False)
    df = df[df.Fitness_value > 0]
    # df = df.drop(by = ['Fitness_value'] < 1)
    print(df)

    # sorted_indices = np.argsort(fitness_values)[::-1]  # Indices of sorted fitness
    sorted_solutions = df['Solution'].tolist()
    sorted_fitness = df['Fitness_value'].tolist()

    # 1️⃣ **Top 20% Elitism**
    retain_count = max(1, int(len(sorted_solutions) * retain_ratio))
    selected_solutions = sorted_solutions[:retain_count]
    print(len(selected_solutions))

    # # 2️⃣ **Probabilistic Selection (Roulette Wheel)**
    remaining_solutions = sorted_solutions[retain_count:]
    remaining_fitness = sorted_fitness[retain_count:]
    if len(remaining_fitness) > 0:  # Avoid division by zero
        fitness_probs = remaining_fitness / np.sum(remaining_fitness)  # Normalize probabilities
        prob_count = max(1, int(len(sorted_solutions) * prob_ratio))
        chosen_indices = np.random.choice(len(remaining_solutions), size=min(prob_count, len(remaining_solutions)),
                                          p=fitness_probs, replace=False)
        selected_solutions += [remaining_solutions[i] for i in chosen_indices]
    print(len(selected_solutions))

    # 3️⃣ **Diversity Selection (Random)**
    diverse_count = min(diverse_count, len(remaining_solutions))
    selected_solutions += random.sample(remaining_solutions, diverse_count)
    print(len(selected_solutions))
    # return selected_solutions
    file = 'out.bson'
    save_solutions(file,selected_solutions)


def load_solutions(filename):
    """Load list of solutions from a BSON file."""
    with open(filename, "rb") as f:
        data = bson.decode(f.read())
    return data["solutions"]

def save_solutions(filename,master_populations):
    """Save list of solutions to a BSON file."""
    with open(filename, "wb") as f:
        f.write(bson.encode({"solutions": master_populations}))
    # with open('populations', 'wb') as fp:
    #     pickle.dump(master_populations, fp)
    # return


def write_to_file(outfile, text):
    if os.path.exists("output.txt"):
        os.remove("output.txt")
    with open(outfile, "a") as file:
        file.write(text + "\n")

def process_population(master_populations, G, crossover_nodes, mutation_nodes, outfile):
    global iteration
    with tqdm(total=stoppping_criteria, desc="Genetic Algorithm Progress", dynamic_ncols=True) as pbar:

        while len(master_populations) <= stoppping_criteria:
            # while len(master_populations) < stopping_criteria and iteration < max_iterations:

            iteration += 1
            # print(f"Iteration: {iteration}, Population Size: {len(master_populations)}")

            fitness_list = fitness_calculation(master_populations, G)
            offspringsolutions, parent_fitness = crossover(G, fitness_list, crossover_nodes, outfile)
            master_populations = mutation(G, offspringsolutions, parent_fitness, mutation_nodes, outfile)
            pbar.update(1)
            # master_populations (mutation(G, offspringsolutions, parent_fitness, mutation_nodes, outfile))

        print("Reached maximum iterations.")
        print('Number of iteration', iteration)
        print("Number of solutions", len(master_populations))
        select_solutions(master_populations, G)


def process_population_old(master_populations, G,crossover_nodes, mutation_nodes, outfile):
    global iteration
    iteration += 1
    if len(master_populations) >= stoppping_criteria:
        print("Reached maximum iterations.")
        print('Number of iteration',iteration)
        print("Number of solutions",len(master_populations))
        # prunning(master_populations)
        select_solutions(master_populations,G)

        # save_solutions("solutions.bson", master_populations)
        # loaded_solutions = load_solutions("solutions.bson")
        # print(len(loaded_solutions))
        return

    # Perform initialization, crossover, mutation, etc., in order
    fitness_list = fitness_calculation(master_populations, G)
    offspringsolutions, parent_fitness = crossover(G, fitness_list, crossover_nodes,outfile)

    # Call the mutation function after crossover


    # print(iteration)
    master_populations = mutation(G, offspringsolutions, parent_fitness, mutation_nodes,outfile)


    # iteration += 1  # Increment iteration after each process
    # Call the main process function again if not reached the max iterations
    process_population(master_populations, G,crossover_nodes, mutation_nodes,outfile)



def fitness_calculation(master_populations,G):

    # print('fitness_calculation')
    # print('fitness function is modularity of graph')
    # Group nodes by community
    nodes = G.nodes()
    fitness_list = []
    # community_ids = master_solutions[0]
    # print(len(master_populations))
    for solution in master_populations:
        community_dict = defaultdict(list)
        community_ids = solution
        # print(len(community_ids))
        for node, community in zip(nodes, community_ids):
            community_dict[community].append(node)
        # Convert to a list of lists
        grouped_nodes = list(community_dict.values())
        mod = nx.community.modularity(G, grouped_nodes)
        fitness_list.append(mod)
    return fitness_list


def crossover(G, fitness_list, crossover_nodes,outfile):
    # print("Crossover")
    """Performs one-point crossover between two randomly selected solutions."""

    # Get indices instead of fitness values
    sampled_indices = random.sample(range(len(fitness_list)), 4)

    # Retrieve corresponding (index, fitness) pairs
    indexed_fitness = [(i, fitness_list[i]) for i in sampled_indices]

    # Sort the sampled solutions based on fitness value (descending)
    indexed_fitness.sort(key=lambda x: x[1], reverse=True)

    # Pick two parents with the highest fitness values
    index_of_parent1, parent1_fitness = indexed_fitness[0]
    index_of_parent2, parent2_fitness = indexed_fitness[1]
    write_to_file(outfile,f'\nParent fitness: {parent1_fitness}, {parent2_fitness}')
    size_of_graph = G.number_of_nodes()
    single_point_crossover = random.randint(0, size_of_graph - 1)

    # Retrieve parent solutions from master_populations
    parent1 = master_populations[index_of_parent1][:]
    parent2 = master_populations[index_of_parent2][:]

    # Extract and swap sublists
    # no0fnodes = 5
    no0fnodes = floor((int(crossover_nodes) * G.number_of_nodes())/100)
    # print('no0fnodes',no0fnodes)
    list1 = parent1[single_point_crossover:single_point_crossover + no0fnodes]
    list2 = parent2[single_point_crossover:single_point_crossover + no0fnodes]
    parent1[single_point_crossover:single_point_crossover + no0fnodes] = list2
    parent2[single_point_crossover:single_point_crossover + no0fnodes] = list1

    # Generate offspring solutions
    offspringsolutions = [parent1, parent2]

    # print('offspringsolutions', len(parent1), len(parent2))
    offsprings_fitness = fitness_calculation(offspringsolutions, G)
    return offspringsolutions, [parent1_fitness,parent2_fitness]
    # print('Fitness of child After crossover', offsprings_fitness)

    # Pass offspring and parents' fitness for mutation
    # mutation(G, offspringsolutions, [parent1_fitness, parent2_fitness])



def mutation(G, offspringsolutions, parent_fitness,no_of_moves,outfile):
    # print("Mutation")
    size_of_graph = len(offspringsolutions[0])
    # no_of_moves = 5

    no_of_moves = int(no_of_moves)
    # print(no_of_moves)
    for move in range(0,no_of_moves):
        for offspring in offspringsolutions:
            # Pick a random node from each and swap
            index1 = random.randint(0,size_of_graph-1)
            index2 = random.randint(0,size_of_graph-1)
            # print(index1,index2)
            node1 = offspring[index1]
            offspring[index1] = offspring[index2]
            offspring[index2] = node1

    # print(offspringsolutions)
    offsprings_fitness = fitness_calculation(offspringsolutions, G)
    # write_to_file("Your text here")
    write_to_file(outfile,f'Fitness of child After Mutation,{offsprings_fitness}')


    # if min(offsprings_fitness) > min(parent_fitness):
    if (offsprings_fitness[0]) > (parent_fitness[0]):
        # print(f'offspring,{(offsprings_fitness[0])} > parent {(parent_fitness[0])}')
        print(f"Population Size: {len(master_populations)}")
        # print(f'offspring,{min(offsprings_fitness)} > parent {min(parent_fitness)}')
        # min_index = offsprings_fitness.index(min(offsprings_fitness))
        # Keep children and remove the weakest parents
        # population.remove(min(population))  # Remove worst solution
        write_to_file(outfile,"Added")
        master_populations.append(offspringsolutions[0])
        # print(master_populations[min_index])
        # master_populations.append(offspringsolutions[1])
    if (offsprings_fitness[1]) > (parent_fitness[1]):
        # print(f'offspring,{(offsprings_fitness[1])} > parent {(parent_fitness[1])}')
        write_to_file(outfile, "Added")
        master_populations.append(offspringsolutions[1])
        # return master_populations

    return master_populations


def parse_args():
    parser = argparse.ArgumentParser(description="Read File and Process Data")
    # Add a global argument for the input file
    parser.add_argument("--inputdir", type=str, required=True, help="Path to the input directory")
    parser.add_argument("--graphfile", type=str, required=True, help="Path to the input graph")
    parser.add_argument("--crossover_nodes", type=str, required=True, help="Number of nodes in the crossover stage")
    parser.add_argument("--mutation_nodes", type=str, required=True, help="Number of nodes in the mutation stage")
    parser.add_argument("--outputfile", type=str, required=True, help="Output file name")
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    print(args.inputdir)
    print(args.graphfile)
    print(args.crossover_nodes)
    print(args.mutation_nodes)
    print(args.outputfile)
    Graph = nx.read_edgelist(args.graphfile)
    # G = nx.read_weighted_edgelist(inputfile)
    # compute the best partition


    write_to_file(args.outputfile,f'number of nodes {Graph.number_of_nodes()} and number of edges {Graph.number_of_edges()}')
    directory = "/Users/shrabanighosh/github project/Generate-Random-Clusters/output/"
    # directory = args.inputdir
    # G = nx.read_edgelist('/Users/shrabanighosh/PycharmProjects/randomComm/random_graph.edgelist')
    print("standard method label propagation modularity value", nx.community.modularity(G, nx.community.label_propagation_communities(G)))
    print("standard method louvain modularity value", nx.community.modularity(G,nx.community.louvain_communities(G, seed=123)))
    G = nx.read_edgelist(args.graphfile)
    # files = glob.glob(directory + "/*.csv")
    # for file in files:
    #     solution = preprocess.process_solutions(file,G)
    #     master_populations.append(solution)
    # print("Number of initial populations",len(master_populations))
    # if os.path.exists(args.outputfile):
    #     os.remove(args.outputfile)
    # if os.path.exists("populations"):
    #     os.remove("populations")

    # master_populations = load_solutions('out.bson')
    process_population(master_populations,G,args.crossover_nodes,args.mutation_nodes, args.outputfile)

    # with open('outfile', 'rb') as fp:
    #     itemlist = pickle.load(fp)

