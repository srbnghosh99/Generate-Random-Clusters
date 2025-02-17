# Random Community Generation Based on Features

This repository provides a set of random generated communities for a given graph which is not purely random rather tailored with some features. The tool takes an input graph file and applies the selected feature technique to random communities with meaningful structures within the network for intitialization purpose of optimization problem with metaheuristic. 

## Features
- Degree-based communities
- Connected component-based communities
- Influence spread-based communities
- Star nodes-based communities
- Random walk-based communities
- Node similarity-based communities
- Random node sampling-based communities
- Graph cliques finding based communities
- Graph traversal-based communities
- Graph ego networks communities

## Requirements
Make sure you have Python 3 installed along with the required libraries. You can install dependencies using:
```bash
pip install -r requirements.txt
```

## Usage
To run the main script, specify the input graph file, the community detection method, and the output file (if applicable).

### Example Command
```bash
python3 main.py --inputfilename trustnetwork.csv random_node_sampling --outputfilename output_degree.csv
```

### Arguments
- `--inputfilename` : Path to the input graph file (e.g., trustnetwork.csv)
- `--command` : The community detection method to apply (choose from the list below)
- `--outputfilename` : Path to save the output (if required by the method)

### Available Commands
- `degree_based_comm` : Detects communities based on node degree
- `connected_components` : Finds communities using connected components
- `influence_spread` : Identifies communities based on influence spread
- `star_nodes` : Detects communities using star nodes
- `random_walk` : Applies random walk for community detection
- `node_similarity` : Uses node similarity to detect communities
- `random_node_sampling` : Performs random node sampling for community detection
- `graph_traversal` : Uses graph traversal (BFS) for community detection
- `clique_clusters` : Uses Cliques to allocate nodes to communities
- `ego_clusters` : Uses ego network to allocate nodes to communities

## Run all features with script file
```bash
./script.sh
```
## Example Output
For an input graph with edges listed in `trustnetwork.csv`, the tool will process the graph using the selected community detection method and generate the corresponding output file (if applicable).

## Error Handling
If no command is provided, the script will display a message:
```plaintext
No command selected. Use --help for usage information.
```

## Contributions
Feel free to contribute by submitting pull requests, reporting issues, or suggesting improvements.



