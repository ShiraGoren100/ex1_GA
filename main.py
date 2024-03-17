import random
import matplotlib.pyplot as plt
import time

BOARD_SIZE = 8
GENERATION_SIZE = 50
NUM_GENERATIONS = 100
ELITE = 4
MUTATION_PROBABILITY = 0.1
MAX_CONFLICTS = 28
BEST_FITNESS = MAX_CONFLICTS


def num_of_conflicts(board):
    conflicts = 0
    for i in range(BOARD_SIZE):
        for j in range(i + 1, BOARD_SIZE):
            if board[i] == board[j] or abs(i - j) == abs(board[i] - board[j]):
                conflicts += 1
    return conflicts


def fitness(chromosome):
    return MAX_CONFLICTS - num_of_conflicts(chromosome)


# Select parents for crossover, using proportioned selection
def selection(population, fitness_vals):
    # Calculate total fitness sum of the population
    total_fitness = sum(fitness for fitness in fitness_vals)
    # Calculate probabilities for each individual based on fitness
    probabilities = [fitness / total_fitness for fitness in fitness_vals]
    parents = []
    pairs_num = (GENERATION_SIZE - ELITE)//2
    for p in range(pairs_num):
        # Select an individual using proportional selection
        selected_index1 = random.choices(range(len(population)), weights=probabilities)[0]
        parent_1 = population[selected_index1]
        selected_index2 = random.choices(range(len(population)), weights=probabilities)[0]
        parent_2 = population[selected_index2]
        parents.append((parent_1, parent_2))
    return parents


# Single-point crossover
def crossover(parent1, parent2):
    crossover_point = random.randint(1, BOARD_SIZE - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


# Mutation operation: switch a value in the chromosome (to value in the range [0,BOARD_SIZE-1])
def mutate(chromosome):
    random_number = random.random()
    # Check if the random number is less than or equal to the mutation probability
    if random_number <= MUTATION_PROBABILITY:
        mutated_position = random.randint(0, BOARD_SIZE - 1)
        mutated_value = random.randint(0, BOARD_SIZE - 1)
        chromosome[mutated_position] = mutated_value
    return chromosome


# Generate the first generation
def generate_random_population():
    population = []
    for i in range(GENERATION_SIZE):
        chrom = [random.randint(0, BOARD_SIZE - 1) for i in range(BOARD_SIZE)]
        population.append(chrom)
    return population


def elitism_helper(fitness_scores):
    # Sort the indices based on the fitness scores
    sorted_indices = sorted(range(len(fitness_scores)), key=lambda index: fitness_scores[index])
    # Get the indices of the chromosomes with the best and worst fitness scores
    best_indices = sorted_indices[-ELITE:]
    worst_indices = sorted_indices[:ELITE]
    return best_indices, worst_indices


def eight_queens_GA():
    gen_solution_found = NUM_GENERATIONS + 1
    start_time = time.time()
    best_fitness = []
    avg_fitness = []
    # Initialize population
    population = generate_random_population()
    gen = 0
    # Do for each generation
    while gen < NUM_GENERATIONS:
        fitness_values = []
        new_gen = []
        # evaluate fitness for each chromosome
        for chrom in population:
            fitness_val = fitness(chrom)
            fitness_values.append(fitness_val)
            if fitness_val == BEST_FITNESS:
                if gen < gen_solution_found:
                    gen_solution_found = gen
                    total_time = time.time() - start_time
                    print(f"eight_queens_GA - solution found at generation {gen}. time: {total_time:.5f}")
        avg_fitness.append(sum(fitness_values) / len(fitness_values))
        # new gen:
        # 1) elitism
        best_chromosomes, worst_chromosomes = elitism_helper(fitness_values)
        best_fitness.append(fitness_values[best_chromosomes[-1]])
        # Add the two best chromosomes to the new gen
        for good_i in best_chromosomes:
            new_gen.append(population[good_i])
        # Remove the two worst chromosomes from the current gen
        worst_chromosomes.sort(reverse=True)  # Sorting in reverse order to avoid index issues
        for bad_i in worst_chromosomes:
            population.pop(bad_i)
            fitness_values.pop(bad_i)
        # 2) selection
        parents = selection(population, fitness_values)
        # 3) crossover
        for (parent1, parent2) in parents:
            child_1, child_2 = crossover(parent1, parent2)
            # 4) mutation
            child_1 = mutate(child_1)
            child_2 = mutate(child_2)
            # add children to new generation
            new_gen.append(child_1)
            new_gen.append(child_2)
        # move to next gen
        population = list(new_gen)
        gen += 1

    # Create the plot
    # plt.plot(range(gen), avg_fitness, label=avg_fitness)
    # plt.plot(range(gen), best_fitness, label=best_fitness)
    # plt.xticks(range(NUM_GENERATIONS))  # Set x-axis ticks to display only whole numbers
    # plt.show()


def eight_queens_random_sol():
    start_time = time.time()
    # find a random solution to 8 queens problem
    num_of_tries = 0
    while True:
        board = [random.randint(0, BOARD_SIZE - 1) for i in range(BOARD_SIZE)]
        if num_of_conflicts(board) == 0:
            total_time = time.time() - start_time
            print(f"eight_queens_random_sol time: {total_time:.5f}")
            return board, num_of_tries


def print_board(board):
    n = len(board)
    for i in range(n):
        for j in range(n):
            if j == board[i]:
                print("Q", end=" ")
            else:
                print(".", end=" ")
        print()
    print()


def main():
    eight_queens_random_sol()
    eight_queens_GA()


if __name__ == "__main__":
    main()