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
import time

master_populations = []

iteration = 0

def select_solutions(master_populations,G,outfile, retain_ratio=0.2, prob_ratio=0.6, diverse_count=10):
    start_time = time.time()
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
    outfile = os.path.splitext(outfile)[0]
    file = outfile + '.bson'
    save_solutions(file,selected_solutions)
    end_time = time.time()
    duration = end_time - start_time
    duration = duration/60
    print(f"The Selection population function took {duration:.2f} minutes to run.")


def load_solutions(filename):
    """Load list of solutions from a BSON file."""
    with open(filename, "rb") as f:
        data = bson.decode(f.read())
    return data["solutions"]

def save_solutions(filename,master_populations):
    """Save list of solutions to a BSON file."""

    with open(filename, "wb") as f:
        f.write(bson.encode({"solutions": master_populations}))



def write_to_file(outfile, text):
    with open(outfile, "a") as file:
        file.write(text + "\n")

def process_population(master_populations, G, fitness_list, crossover_nodes, mutation_nodes, outfile,popusize,max_iterations):
    global iteration
    start_time = time.time()
    popusize = int(popusize)
    max_iterations = int(max_iterations)


    with tqdm(total=max_iterations, desc="Genetic Algorithm Progress", dynamic_ncols=True) as pbar:
        while len(master_populations) <= popusize:

            iteration += 1
            if iteration == max_iterations:
                print("Reached maximum iteration, no solution found.")
                break
            offspringsolutions, parent_fitness = crossover(G, fitness_list, crossover_nodes,outfile)
            master_populations, fitness_list = mutation(G, offspringsolutions, fitness_list, parent_fitness, mutation_nodes,outfile)

            pbar.update(1)


        print("Reached maximum solutions.")
        print('Number of iteration', iteration)
        print("Number of solutions", len(master_populations))
        end_time = time.time()
        duration = end_time - start_time
        duration = duration / 60
        print(f"The Processing population took {duration:.2f} minutes to run.")
        select_solutions(master_populations, G,outfile)


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
    # write_to_file(outfile,f'\nParent fitness: {parent1_fitness}, {parent2_fitness}')
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


    return offspringsolutions, [parent1_fitness,parent2_fitness]




def mutation(G, offspringsolutions, fitness_list, parent_fitness,no_of_moves,outfile):
    # print("Mutation")
    size_of_graph = len(offspringsolutions[0])

    no_of_moves = int(no_of_moves)
    for move in range(0,no_of_moves):
        for offspring in offspringsolutions:
            # Pick a random node from each and swap
            index1 = random.randint(0,size_of_graph-1)
            index2 = random.randint(0,size_of_graph-1)
            node1 = offspring[index1]
            offspring[index1] = offspring[index2]
            offspring[index2] = node1

    offsprings_fitness = fitness_calculation(offspringsolutions, G)
    if offsprings_fitness[0]> .571136 or offsprings_fitness[1]> .571136:
        print("Found Solution")
        master_populations.append(offsprings_fitness[0])
        master_populations.append(offsprings_fitness[1])
        write_to_file(outfile,f'Found Solution')
        return


    if (offsprings_fitness[0]) > (parent_fitness[0]):
        write_to_file(outfile,f'offspring,{(offsprings_fitness[0])} > parent {(parent_fitness[0])}')
        master_populations.append(offspringsolutions[0])
        fitness_list.append(offsprings_fitness[0])
    if (offsprings_fitness[1]) > (parent_fitness[1]):
        write_to_file(outfile, f'offspring,{(offsprings_fitness[1])} > parent {(parent_fitness[1])}')
        master_populations.append(offspringsolutions[1])
        fitness_list.append(offsprings_fitness[1])
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
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    print(args.inputdir)
    print(args.flag)
    print(args.graphfile)
    print(args.crossover_nodes)
    print(args.mutation_nodes)
    print(args.outputfile)
    print(args.populationsize)
    print(args.iteration)
    Graph = nx.read_edgelist(args.graphfile)

    write_to_file(args.outputfile,f'number of nodes {Graph.number_of_nodes()} and number of edges {Graph.number_of_edges()}')
    directory = args.inputdir

    G = nx.read_edgelist(args.graphfile)
    if args.flag == '1':
        files = glob.glob(directory + "/*.csv")
        for file in files:
            solution = preprocess.process_solutions(file,G)
            master_populations.append(solution)
    if args.flag == '2':
        master_populations = load_solutions(args.inputdir)
    print("Number of initial populations",len(master_populations))
    if os.path.exists(args.outputfile):
        os.remove(args.outputfile)

    if os.path.exists(args.outputfile):
        os.remove(args.outputfile)
    fitness_list = fitness_calculation(master_populations, G)
    print('fitness_list',len(fitness_list))
    write_to_file(args.outputfile,f'standard method label propagation modularity value, {nx.community.modularity(G, nx.community.label_propagation_communities(G))}')
    write_to_file(args.outputfile,f'standard method louvain modularity value,{nx.community.modularity(G, nx.community.louvain_communities(G, seed=123))}')
    process_population(master_populations,G,fitness_list,args.crossover_nodes,args.mutation_nodes, args.outputfile,args.populationsize,args.iteration)

