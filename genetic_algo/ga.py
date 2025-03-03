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


master_populations = []
global iteration
stoppping_criteria = 10


def process_population(master_populations, G):
    # global iteration
    if len(master_populations) >= stoppping_criteria:
        print("Reached maximum iterations.")
        return

    # Perform initialization, crossover, mutation, etc., in order
    fitness_list = fitness_calculation(master_populations, G)
    offspringsolutions, parent_fitness = crossover(G, fitness_list, no0fnodes=5)

    # Call the mutation function after crossover
    master_populations = mutation(G, offspringsolutions, parent_fitness)


    # iteration += 1  # Increment iteration after each process
    # Call the main process function again if not reached the max iterations
    process_population(master_populations, G)


def initialization(master_populations, G):
    # Initialize the process, which can call crossover and mutation
    process_population(master_populations, G)




def write_to_file(text):
    # if os.path.exists("output.txt"):
    #     os.remove("output.txt")
    with open("output.txt", "a") as file:
        file.write(text + "\n")

def initialization(master_populations,G):
    print("initialization")
    fitness_list = fitness_calculation(master_populations,G)
    # crossover(G, fitness_list, no0fnodes = 5)

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


def crossover(G, fitness_list, no0fnodes):
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
    write_to_file(f'\nParent fitness: {parent1_fitness}, {parent2_fitness}')
    size_of_graph = G.number_of_nodes()
    single_point_crossover = random.randint(0, size_of_graph - 1)

    # Retrieve parent solutions from master_populations
    parent1 = master_populations[index_of_parent1][:]
    parent2 = master_populations[index_of_parent2][:]

    # Extract and swap sublists
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



def mutation(G, offspringsolutions, parent_fitness):
    # iteration =+ 1
    # print('mutation',iteration)
    # print("Mutation")
    size_of_graph = len(offspringsolutions[0])
    no_of_moves = 2
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
    write_to_file(f'Fitness of child After Mutation,{offsprings_fitness}')


    if min(offsprings_fitness) > min(parent_fitness):
        print(f'{min(offsprings_fitness)} > {min(parent_fitness)}')
        # Keep children and remove the weakest parents
        # population.remove(min(population))  # Remove worst solution
        write_to_file("Added")
        # write_to_file("\n")
        master_populations.append(offspringsolutions[0])
        master_populations.append(offspringsolutions[1])
        return master_populations
    else:
        # Keep parents and discard children
        # write_to_file("Did not Added")
        return master_populations
        # pass  # No change in population

    # if (len(master_populations) == stoppping_criteria):
    #     sys.exit("Terminating program from recursion!")
    # initialization(master_populations, G)


def parse_args():
    parser = argparse.ArgumentParser(description="Read File and Process Data")
    # Add a global argument for the input file
    parser.add_argument("--inputdir", type=str, required=True, help="Path to the input directory")
    parser.add_argument("--graphfile", type=str, required=True, help="Path to the input graph")
    return parser.parse_args()

def parse_args():
    parser = argparse.ArgumentParser(description="Read File and Process Data")
    # Add a global argument for the input file
    parser.add_argument("--inputdir", type=str, required=True, help="Path to the input directory")
    parser.add_argument("--graphfile", type=str, required=True, help="Path to the input graph")
    return parser.parse_args()
    
if __name__ == '__main__':
    master_solutions = []
    args = parse_args()
    print(args.inputdir)
    print(args.graphfile)
    # Graph = nx.read_edgelist(args.graphfile)
    G = nx.read_edgelist(args.graphfile)
    print(Graph.number_of_nodes(), Graph.number_of_edges())
    directory = args.inputdir
    
    # directory = "/Users/shrabanighosh/github project/Generate-Random-Clusters/output/"
    # G = nx.read_edgelist('/Users/shrabanighosh/PycharmProjects/randomComm/random_graph.edgelist')

    args = parse_args()
    print(args.inputdir)
    print(args.graphfile)
    Graph = nx.read_edgelist(args.graphfile)

    write_to_file(f'number of nodes {Graph.number_of_nodes()} and number of edges {Graph.number_of_edges()}')
    directory = "/Users/shrabanighosh/github project/Generate-Random-Clusters/output/"
    # directory = args.inputdir
    G = nx.read_edgelist('/Users/shrabanighosh/PycharmProjects/randomComm/random_graph.edgelist')
    # G = nx.read_edgelist(args.graphfile)
    files = glob.glob(directory + "/*.csv")
    for file in files:
        solution = preprocess.process_solutions(file,G)
        master_populations.append(solution)
    print("Number of populations",len(master_populations))
    if os.path.exists("output.txt"):
        os.remove("output.txt")
    process_population(master_populations,G)

