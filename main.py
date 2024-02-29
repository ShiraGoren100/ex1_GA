import random
import matplotlib.pyplot as plt

NUM_ROWS = 8
MAX_FITNESS = 2
GENERATION_SIZE = 50
NUM_GENERATIONS = 100
ELITE = 2
MUTATION_PROBABILITY = 0.1
CONFLICTS_UPPER_BOUND = 28  # 7+6+5+4+3+2+1 = 28 todo explain


def num_of_conflicts(board):
    conflicts = 0
    for i in range(NUM_ROWS):
        for j in range(i + 1, NUM_ROWS):
            if board[i] == board[j] or abs(i - j) == abs(board[i] - board[j]):
                conflicts += 1
    return conflicts


def fitness(chromosome):
    return CONFLICTS_UPPER_BOUND - num_of_conflicts(chromosome)


# Select parents for crossover using proportioned selection
def selection(population, fitness_vals):
    # select parents for crossover:
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


# Crossover operation (single-point crossover)
def crossover(parent1, parent2):
    crossover_point = random.randint(1, 7)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


# Mutation operation: switch a value in the chromosome (the value is in range 0-7)
def mutate(chromosome):
    random_number = random.random()
    # Check if the random number is less than or equal to the mutation probability
    if random_number <= MUTATION_PROBABILITY:
        mutated_position = random.randint(0, 7)
        mutated_value = random.randint(0, 7)
        chromosome[mutated_position] = mutated_value
    return chromosome


def generate_random_population():
    # generate the first generation
    population = []
    for i in range(GENERATION_SIZE):
        chrom = [random.randint(0, NUM_ROWS - 1) for _ in range(NUM_ROWS)]
        population.append(chrom)
    return population


def elitism_helper(fitness_scores):
    # Sort the indices based on the fitness scores
    sorted_indices = sorted(range(len(fitness_scores)), key=lambda k: fitness_scores[k])

    # Get the indices of the arrays with the best and worst scores
    best_indices = sorted_indices[-ELITE:]
    worst_indices = sorted_indices[:ELITE]

    return best_indices, worst_indices


def eight_queens_GA():
    best_fitness = [] #todo change to avg?
    best_chromosomes = []
    worst_chromosomes = []
    # initialize population
    population = generate_random_population()
    # do for each generation
    gen = 0
    while gen < NUM_GENERATIONS:
        fitness_values = []
        new_gen = []
        # evaluate fitness for each chromosome
        for chrom in population:
            fitness_values.append(fitness(chrom))
        # avg_fitness.append(sum(fitness_values) / len(fitness_values)) todo
        # new gen:
        # 1) elitism
        best_chromosomes, worst_chromosomes = elitism_helper(fitness_values)
        best_chromosomes.sort(reverse=True)
        best_fitness.append(fitness_values[best_chromosomes[0]])
        if fitness_values[best_chromosomes[0]] == CONFLICTS_UPPER_BOUND:
            gen += 1
            break
        for good_i in best_chromosomes:
            new_gen.append(population[good_i])
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

    print(num_of_conflicts(population[best_chromosomes[0]]))
    print_board(population[best_chromosomes[0]])
    print(gen)
    # Create the plot
    plt.plot(range(gen), best_fitness)
    # Set x-axis ticks to display only whole numbers
    plt.xticks(range(NUM_GENERATIONS))
    plt.show()



def eight_queens_random_sol():
    # find a random solution to 8 queens problem
    num_of_tries = 0
    while True:
        board = [random.randint(0, NUM_ROWS-1) for _ in range(NUM_ROWS)]
        if num_of_conflicts(board) == 0:
            # print_board(board)
            return board, num_of_tries


# def print_board(board):
#     n = len(board)
#     for i in range(n):
#         for j in range(n):
#             if board[j] == i:
#                 print("Q", end=" ")
#             else:
#                 print(".", end=" ")
#         print()

def print_board(board):
    n = len(board)
    for i in range(n):
        for j in range(n):
            if j == board[i]:
                print("Q", end=" ")
            else:
                print(".", end=" ")
        print()

def main():
    # eight_queens_random_sol()
    eight_queens_GA()


if __name__ == "__main__":
    main()
