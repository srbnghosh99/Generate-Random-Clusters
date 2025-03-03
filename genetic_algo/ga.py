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


def initialization(master_solutions, G, stop):
    fitness_list = fitness_calculation(master_solutions, G)
    # print(fitness_list)
    stop = stop
    crossover(G, master_solutions, fitness_list, no0fnodes = 5)

def fitness_calculation(master_solutions, G):
    print('fitness_calculation')
    # Group nodes by community
    nodes = G.nodes()
    fitness_list = []
    # community_ids = master_solutions[0]
    for solution in master_solutions:
        community_dict = defaultdict(list)
        community_ids = solution
        for node, community in zip(nodes, community_ids):
            community_dict[community].append(node)
        # Convert to a list of lists
        grouped_nodes = list(community_dict.values())

        mod = nx.community.modularity(G, grouped_nodes)
        fitness_list.append(mod)
        # print(mod)
    return fitness_list


def crossover(G, master_solutions, fitness_list, no0fnodes):
    # stop += 1
    # print('stop', stop)
    """Performs one-point crossover between two randomly selected solutions."""
    # if len(communities) < 2:
    #     print("Not enough communities for crossover.")
    #     return communities


    # randomly select 4 solution
    idx1, idx2,idx3,idx4 = random.sample(fitness_list, 4)
    # pick two parent with higher fitness value
    parent1_fitness = idx3 if idx3 > idx1 else idx1
    parent2_fitness = idx2 if idx2 > idx4 else idx4
    print("Fitness of parents ",parent1_fitness,parent2_fitness)

    # find index of selected fitness value
    index_of_parent1 = fitness_list.index(parent1_fitness)
    index_of_parent2 = fitness_list.index(parent2_fitness)
    size_of_graph = G.number_of_nodes()
    single_point_crossover = random.randint(0,size_of_graph)

    # find
    parent1 = master_solutions[index_of_parent1]
    parent2 = master_solutions[index_of_parent2]
    # Extract sublists from parents
    list1 = parent1[single_point_crossover:single_point_crossover+no0fnodes]
    list2 = parent2[single_point_crossover:single_point_crossover+no0fnodes]
    # Swap the sublists
    parent1[single_point_crossover:single_point_crossover + no0fnodes] = list2
    parent2[single_point_crossover:single_point_crossover + no0fnodes] = list1

    offspringsolutions = []
    offspringsolutions.append(parent1)
    offspringsolutions.append(parent2)
    offsprings_fitness = fitness_calculation(offspringsolutions,G)
    print('Fitness of child After crossover',offsprings_fitness)
    mutation(offspringsolutions)


def mutation(offspringsolutions):
    print('mutation')
    size_of_graph = len(offspringsolutions[0])
    no_of_moves = 2
    for move in range(0,no_of_moves):
        for offspring in offspringsolutions:
            # Pick a random node from each and swap
            index1 = random.randint(0,size_of_graph)
            index2 = random.randint(0,size_of_graph)
            node1 = offspring[index1]
            offspring[index1] = offspring[index2]
            offspring[index2] = node1

    # print(offspringsolutions)
    offsprings_fitness = fitness_calculation(offspringsolutions, G)
    print('Fitness of child After Mutation',offsprings_fitness)

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
    Graph = nx.read_edgelist(args.graphfile)

    print(Graph.number_of_nodes(), Graph.number_of_edges())
    directory = args.inputdir
    G = nx.read_edgelist(args.graphfile)
    
    # directory = "/Users/shrabanighosh/github project/Generate-Random-Clusters/output/"
    # G = nx.read_edgelist('/Users/shrabanighosh/PycharmProjects/randomComm/random_graph.edgelist')
    files = glob.glob(directory + "/*.csv")
    for file in files:
        solution = preprocess.process_solutions(file,G)
        master_solutions.append(solution)
    print("Number of populations",len(master_solutions))
    initialization(master_solutions, G, stop=0)

