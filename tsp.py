import math

import random
import matplotlib.pyplot as plt

MAX_FITNESS = 2
GENERATION_SIZE = 50
NUM_GENERATIONS = 100
ELITE = 2
MUTATION_PROBABILITY = 0.1
NUM_CITIES = 5
MIN_INT = -10
MAX_INT = 10


def euclidean_distance(p1, p2):
    """Calculate the Euclidean distance between two cities."""
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def fitness(chromosome):
    path_cost = 0
    for i in range(NUM_CITIES - 1):
        path_cost += euclidean_distance(chromosome[i], chromosome[i+1])
    return 1 / path_cost


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


# Crossover operation
# todo call (parent1, parent2), (parent2, parent1)
def crossover(parent1, parent2):
    crossover_point = random.randint(1, NUM_CITIES - 1)

    child1 = parent1[:crossover_point]
    unvisited = parent1[crossover_point:]
    child1 += [city for city in parent2[1:] if city in unvisited]

    child2 = parent2[:crossover_point]
    unvisited = parent2[crossover_point:]
    child2 += [city for city in parent1[1:] if city in unvisited]

    return child1, child2


# Mutation operation: switch a value in the chromosome
def mutate(chromosome):
    random_number = random.random()
    # Check if the random number is less than or equal to the mutation probability
    if random_number <= MUTATION_PROBABILITY:
        index1 = random.randint(1, NUM_CITIES - 1)
        index2 = random.randint(1, NUM_CITIES - 1)
        chromosome[index1], chromosome[index2] = chromosome[index2], chromosome[index1]
    return chromosome


def generate_map():
    map = set()
    while len(map) < NUM_CITIES - 1:
        map.add((random.randint(MIN_INT, MAX_INT), random.randint(MIN_INT, MAX_INT)))
    return map

def generate_random_population(map, start_point):
    # generate the first generation
    map.remove(start_point)
    population = []
    for i in range(GENERATION_SIZE):
        chrom = list(map)
        random.shuffle(chrom)
        chrom.insert(0, start_point)
        chrom.append(start_point)
        population.append(chrom)
    return population


def elitism_helper(fitness_scores):
    # Sort the indices based on the fitness scores
    sorted_indices = sorted(range(len(fitness_scores)), key=lambda k: fitness_scores[k])

    # Get the indices of the arrays with the best and worst scores
    best_indices = sorted_indices[-ELITE:]
    worst_indices = sorted_indices[:ELITE]

    return best_indices, worst_indices


def tsp_GA(map, start_point):
    best_fitness = [] #todo change to avg?
    # best_chromosomes = [] todo
    # worst_chromosomes = []
    # initialize population
    population = generate_random_population(map, start_point)
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
        if gen == NUM_GENERATIONS-1:
            print(population[best_chromosomes[0]])
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

    print()
    # Create the plot
    plt.plot(range(gen), best_fitness)
    # Set x-axis ticks to display only whole numbers
    plt.xticks(range(NUM_GENERATIONS))
    plt.show()


def greedy_tsp(cities, start):
    """Greedy algorithm for the Traveling Salesman Problem."""
    n = len(cities)
    unvisited = set(range(n))
    path = [start]
    current_city = start
    unvisited.remove(start)

    while unvisited:
        closest_city = min(unvisited, key=lambda city: euclidean_distance(cities[current_city], cities[city]))
        path.append(closest_city)
        unvisited.remove(closest_city)
        current_city = closest_city

    # Return to the start city to complete the path
    path.append(start)
    return path

# Example cities represented as (x, y) coordinates
cities = [(0, 0), (1, 2), (3, 1), (2, 3), (1, 1)]
start_city = 0  # Start from city A

# Find the TSP tour using the greedy algorithm
#tsp_tour = greedy_tsp(cities, start_city)
#print("TSP Tour:", tsp_tour)

map = generate_map()
start_point = random.choice(tuple(map))
list_map = list(map)
tsp_tour = greedy_tsp(list_map, list_map.index(start_point))
print(tsp_tour)
tsp_GA(map, start_point)



