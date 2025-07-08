import pandas as pd
import time
import pandas as pd
import random
import numpy as np
import file_read_write

def select_solutions(master_populations,fitness_list,G,outfile, retain_ratio=0.2, prob_ratio=0.6, diverse_count=10):
    # start_time = time.time()
    # MAX_SOLUTIONS = 100000
    df = pd.DataFrame({'Solution':master_populations,'Fitness_value':fitness_list})
    df = df.sort_values(by = ['Fitness_value'],ascending = False)
    df = df[df.Fitness_value > 0]
    # print(df)

    # sorted_indices = np.argsort(fitness_values)[::-1]  # Indices of sorted fitness
    sorted_solutions = df['Solution'].tolist()
    sorted_fitness = df['Fitness_value'].tolist()

    # Elitism
    retain_count = max(1, int(len(sorted_solutions) * retain_ratio))
    selected_solutions = sorted_solutions[:retain_count]
    selected_fitness = sorted_fitness[:retain_count]
    print(len(selected_solutions))

    # Probabilistic Selection (Roulette Wheel)
    remaining_solutions = sorted_solutions[retain_count:]
    remaining_fitness = sorted_fitness[retain_count:]

    if len(remaining_fitness) > 0:
        fitness_probs = np.array(remaining_fitness) / np.sum(remaining_fitness)
        prob_count = max(1, int(len(sorted_solutions) * prob_ratio))
        chosen_indices = np.random.choice(len(remaining_solutions), size=min(prob_count, len(remaining_solutions)),
                                          p=fitness_probs, replace=False)
        selected_solutions += [remaining_solutions[i] for i in chosen_indices]
        selected_fitness += [remaining_fitness[i] for i in chosen_indices]
    print(len(selected_solutions))

    # Diversity Selection (Random)
    remaining_indices = list(set(range(len(remaining_solutions))) - set(chosen_indices))
    if len(remaining_indices) >= diverse_count:
        random_indices = random.sample(remaining_indices, diverse_count)
        selected_solutions += [remaining_solutions[i] for i in random_indices]
        selected_fitness += [remaining_fitness[i] for i in random_indices]
    print(len(selected_solutions))
    # if len(selected_solutions) > MAX_SOLUTIONS:
    #     selected_solutions = selected_solutions[:100000]
    #     selected_fitness = selected_fitness[:100000]
    combined = [{"solution": sol, "fitness": fit} for sol, fit in zip(selected_solutions, selected_fitness)]
    file_read_write.write_to_file(outfile, f'Number of solutions selected: {len(combined)}')

    print("Number of selected solutions", len(combined))
    MAX_SOLUTIONS = 100000
    if len(combined) > MAX_SOLUTIONS:
        combined = random.sample(combined, 100000)
        # combined = combined[:100000]

    # end_time = time.time()
    # duration = end_time - start_time
    # duration = duration/60
    # print(f"The Selection population function took {duration:.2f} minutes to run.")
    return  combined



