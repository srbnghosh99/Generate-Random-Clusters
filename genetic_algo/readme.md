Generate non-overlapping community populations based on features, randomization and previous bson file based populations for a given graph.

### Arguments
- `--inputdir` : Path to the input graph file (e.g., trustnetwork.csv)
- `--graphfile` : The community detection method to apply (choose from the list below)
- `--flag` : Flag value options: 1,2,3. 1 for feature based, 2 for intial randomization and 3 for existing bson file initialization
- `--crossover_nodes` : Percentage of nodes want to be swapped across different populations
- `--mutation_nodes` : Number of ndoes want to be swapped at the mutations stage
- `--populationsize` : Number of solutions
- `--iteration` : Number of iterations
- `--function` : Optimization function of genetic algorithm


## Run code with script file
```bash
./script.sh
```
