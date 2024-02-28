import random


NUM_ROWS = 8
MAX_FITNESS = 2
GENERATION_SIZE = 10
GENERATIONS = 20


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
def selection(population):
    # select parents for crossover


# Crossover operation (single-point crossover)
def crossover(parent1, parent2):
    crossover_point = random.randint(1, 7)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child

# Mutation operation (swap two positions)
def mutate(board_state):
    pos1, pos2 = random.sample(range(8), 2)
    board_state[pos1], board_state[pos2] = board_state[pos2], board_state[pos1]
    return board_state

# Generate the initial population
population = [(generate_board_state(), 0) for _ in range(POPULATION_SIZE)]

# Main Genetic Algorithm loop
for generation in range(MAX_GENERATIONS):
    # Calculate fitness for each board state
    population = [(board_state, calculate_fitness(board_state)) for board_state, _ in population]

    # Check if solution is found
    best_board_state = max(population, key=lambda x: x[1])[0]
    if calculate_fitness(best_board_state) == 28:
        print("Solution found in generation", generation)
        break

    # Create the next generation
    new_population = []

    # Elitism: Keep the best board state from the previous generation
    new_population.append(max(population, key=lambda x: x[1]))

    # Perform selection, crossover, and mutation
    while len(new_population) < POPULATION_SIZE:
        parent1 = selection(population)
        parent2 = selection(population)
        child = crossover(parent1[0], parent2[0])
        if random.random() < MUTATION_RATE:
            child = mutate(child)
        new_population.append((child, 0))

    # Update the population
    population = new_populatio


def generate_randon_population():
    population = []
    for i in range(GENERATION_SIZE):
        chrom = [random.randint(0, NUM_ROWS - 1) for _ in range(NUM_ROWS)]
        population.append(chrom)
    return population


def elitism_helper(population, fitness_scores):
    # Sort the scores and arrays based on the scores
    sorted_indices = sorted(range(len(fitness_scores)), key=lambda k: fitness_scores[k])

    # Get the indices of the arrays with the best and worst scores
    best_indices = sorted_indices[-2:]
    worst_indices = sorted_indices[:2]

    # Get the arrays corresponding to the best and worst scores
    best_chromosomes= [population[i] for i in best_indices]
    worst_chromosomes = [population[i] for i in worst_indices]
    return best_chromosomes, worst_chromosomes

def eight_queens_GA():
    # intialize population
    population = generate_randon_population()
    # do for each generation
    for gen in range(GENERATIONS):
        fitness_values = []
        new_gen = []
        # evaluate fitness for each chromosome
        for chrom in population:
            fitness_values.append(fitness(chrom))
        #new gen:
        # 1) elitism
        best_chromosomes, worst_chromosomes = elitism_helper(population, fitness_values)
        for good_c in best_chromosomes:
            new_gen.append(good_c)
        for bad_c in worst_chromosomes:
            population.remove(bad_c)

        # 2) selection

        # 3) crossover
        # 4) mutation




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
