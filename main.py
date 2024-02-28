import random


NUM_ROWS = 8
MAX_FITNESS = 2
GENERATION_SIZE = 10
GENERATIONS = 20
ELITE = 2
MUTATION_PROBABILITY = 0.001

def num_of_conflicts(board):
    conflicts = 0
    for i in range(NUM_ROWS):
        for j in range(i + 1, NUM_ROWS):
            if board[i] == board[j] or abs(i - j) == abs(board[i] - board[j]):
                conflicts += 1
    return conflicts


def fitness(board):
    # return high fitness for fewer conflicts in positions
    score = num_of_conflicts(board)
    if score == 0:
        return MAX_FITNESS
    return 1/score



# Select parents for crossover using proportioned selection
def selection(population, fitness_vals):
    # select parents for crossover

    # Calculate total fitness sum of the population
    total_fitness = sum(fitness for fitness in fitness_vals)
    # Calculate probabilities for each individual based on fitness
    probabilities = [fitness / total_fitness for fitness in fitness_vals]
    parents = []
    pairs_num = (GENERATION_SIZE - ELITE)//2
    for g in range(pairs_num):
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


# Mutation operation: switch a value in the chromosome in valid range
def mutate(chromosome):
    random_number = random.random()
    # Check if the random number is less than or equal to the probability
    if random_number <= MUTATION_PROBABILITY:
        mutated_position = random.randint(0, 8)
        mutated_value = random.randint(0, 8)
        chromosome[mutated_position] = mutated_value

    return chromosome


def generate_randon_population():
    # generate the first generation
    population = []
    for i in range(GENERATION_SIZE):
        chrom = [random.randint(0, NUM_ROWS - 1) for _ in range(NUM_ROWS)]
        population.append(chrom)
    return population


def elitism_helper(population, fitness_scores):
    # Sort the scores and arrays based on the scores
    sorted_indices = sorted(range(len(fitness_scores)), key=lambda k: fitness_scores[k])

    # Get the indices of the arrays with the best and worst scores
    best_indices = sorted_indices[-ELITE:]
    worst_indices = sorted_indices[:ELITE]

    # # Get the arrays corresponding to the best and worst scores
    # best_chromosomes= [population[i] for i in best_indices]
    # worst_chromosomes = [population[i] for i in worst_indices]
    return best_indices, worst_indices

def eight_queens_GA():
    # intialize population
    population = generate_randon_population()
    # do for each generation
    for gen in range(GENERATIONS):
        fitness_values = []
        new_gen = []
        children = []
        # evaluate fitness for each chromosome
        for chrom in population:
            fitness_values.append(fitness(chrom))
        #new gen:
        # 1) elitism
        best_chromosomes, worst_chromosomes = elitism_helper(population, fitness_values)
        for good_i in best_chromosomes:
            new_gen.append(population[good_i])
        for bad_i in worst_chromosomes:
            population.remove(population[bad_i])
            fitness_values.remove(fitness_values[bad_i])
        # 2) selection
        parents = selection(population, fitness_values)
        # 3) crossover
        for (parent1, parent2) in parents:
            child_1, child_2 = crossover(parent1, parent2)
            # 4) mutation in low probability

            #add to new generation




def eight_queens_random_sol():
    # find a random solution to 8 queens problem
    while True:
        board = [random.randint(0, NUM_ROWS-1) for _ in range(NUM_ROWS)]
        if num_of_conflicts(board) == 0:
            # print_board(board)
            return board


def print_board(board):
    n = len(board)
    for i in range(n):
        for j in range(n):
            if board[j] == i:
                print("Q", end=" ")
            else:
                print(".", end=" ")
        print()

def main():
    eight_queens_random_sol()




if __name__ == "__main__":
    main()
