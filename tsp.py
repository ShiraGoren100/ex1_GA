import itertools
import math
import time
import random
import matplotlib.pyplot as plt

GENERATION_SIZE = 200
NUM_GENERATIONS = 1000
ELITE = 4
MUTATION_PROBABILITY = 0.01
CROSSOVER_PROBABILITY = 0.7
NUM_CITIES = 20
MIN_INT = -10
MAX_INT = 10
TEST_MAP = [(-2, 8), (9, -7), (-3, -3), (-6, -2), (7, 1), (-1, 4), (-1, 7),
            (-5, -4), (5, 3), (6, -8), (-7, 4), (-2, -3), (-1, 6), (8, -5),
            (-6, 2), (7, -10), (1, -2), (6, -3), (-4, -4), (6, 9)]

RESULTS_FILE = "207814989_209540731.txt"


def euclidean_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def path_cost(path, cities_map):
    path_cost = 0
    for i in range(NUM_CITIES - 1):
        path_cost += euclidean_distance(cities_map[path[i]], cities_map[path[i + 1]])
    return path_cost


def fitness(chromosome, cities_map):
    return 1 / path_cost(chromosome, cities_map)


# Select parents for crossover using proportioned selection
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


# Crossover operation
def crossover(parent1, parent2):
    random_number = random.random()
    # Check if the random number is less than or equal to the crossover probability
    if random_number <= CROSSOVER_PROBABILITY:
        crossover_point = random.randint(1, NUM_CITIES - 1)

        child1 = parent1[:crossover_point]
        unvisited = parent1[crossover_point:]
        child1 += [city for city in parent2[1:] if city in unvisited]

        child2 = parent2[:crossover_point]
        unvisited = parent2[crossover_point:]
        child2 += [city for city in parent1[1:] if city in unvisited]

        return child1, child2

    return parent1.copy(), parent2.copy()


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
    while len(map) < NUM_CITIES:
        map.add((random.randint(MIN_INT, MAX_INT), random.randint(MIN_INT, MAX_INT)))
    return list(map)


def generate_map_from_file(file):
    cities_map = []
    with open(file, "r") as file:
        for line in file:
            x, y = map(int, line.split())
            cities_map.append((x, y))
    return cities_map


def generate_random_population(start_point):
    population = []
    for i in range(GENERATION_SIZE):
        chrom = [start_point]
        unvisited = [i for i in range(NUM_CITIES) if i != start_point]
        while unvisited:
            city = random.choice(unvisited)
            chrom.append(city)
            unvisited.remove(city)
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


def GA_tsp(cities_map, start):
    start_time = time.time()
    result = []
    best_fitness = []
    avg_fitness = []
    # initialize population
    population = generate_random_population(start)
    # do for each generation
    gen = 0
    while gen < NUM_GENERATIONS:
        fitness_values = []
        new_gen = []
        # evaluate fitness for each chromosome
        for chrom in population:
            fitness_values.append(fitness(chrom, cities_map))
        avg_fitness.append(sum(fitness_values) / len(fitness_values))
        # new gen:
        # 1) elitism
        best_chromosomes, worst_chromosomes = elitism_helper(fitness_values)
        best_fitness.append(fitness_values[best_chromosomes[-1]])
        if gen == NUM_GENERATIONS - 1:
            result = population[best_chromosomes[1]]
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

    total_time = time.time() - start_time
    print(f"time: {total_time:.5f}")

    # Create the plot
    plt.plot(range(gen), avg_fitness, label=avg_fitness)
    plt.plot(range(gen), best_fitness, label=best_fitness)
    plt.xticks(range(NUM_GENERATIONS))  # Set x-axis ticks to display only whole numbers
    plt.show()
    return result


def greedy_tsp(cities_map, start):
    unvisited = set(range(NUM_CITIES))
    path = [start]
    current_city = start
    unvisited.remove(start)

    while unvisited:
        closest_city = min(unvisited, key=lambda city: euclidean_distance(cities_map[current_city], cities_map[city]))
        path.append(closest_city)
        unvisited.remove(closest_city)
        current_city = closest_city

    # Return to the start city to complete the path
    path.append(start)
    return path


def brute_force_tsp(cities_map, start):
    min_cost = float('inf')
    optimal_path = []
    cities_minus_start = [i for i in range(NUM_CITIES) if i != start]
    # Generate all possible permutations of cities
    for perm in itertools.permutations(cities_minus_start):
        path = [start] + list(perm) + [start]
        current_cost = path_cost(path, cities_map)
        if current_cost < min_cost:
            min_cost = current_cost
            optimal_path = path
    return optimal_path


def print_to_results_file(path):
    with open(RESULTS_FILE, "w") as file:
        for city in path:
            file.write(f"{city + 1}\n")


def run_from_file(file):
    cities_map = generate_map_from_file(file)
    start_city = 0

    print("GA results:")
    GA_path = GA_tsp(cities_map, start_city)
    print("path cost: " + str(path_cost(GA_path, cities_map)))

    print()

    print("greedy algorithm results:")
    start_time = time.time()
    greedy_path = greedy_tsp(cities_map, start_city)
    total_time = time.time() - start_time
    print(f"time: {total_time:.5f}")
    print("path cost: " + str(path_cost(greedy_path, cities_map)))

    print_to_results_file(GA_path)


def run_all_algorithms(cities_map, start_city):

    # print("brute force results:")
    # start_time = time.time()
    # optimal_path = brute_force_tsp(cities_map, start_city)
    # total_time = time.time() - start_time
    # print(f"time: {total_time:.5f}")
    # print(optimal_path)
    # print("path cost: " + str(path_cost(optimal_path, cities_map)))
    #
    # print()

    print("greedy algorithm results:")
    start_time = time.time()
    greedy_path = greedy_tsp(cities_map, start_city)
    total_time = time.time() - start_time
    print(f"time: {total_time:.5f}")
    print(greedy_path)
    print("path cost: " + str(path_cost(greedy_path, cities_map)))

    print()

    print("GA results:")
    GA_path = GA_tsp(cities_map, start_city)
    print(GA_path)
    print("path cost: " + str(path_cost(GA_path, cities_map)))


def main():
    for i in range(5):
        print()
        print(i)
        run_all_algorithms(TEST_MAP, 0)

    # run_from_file("tsp.txt")


if __name__ == "__main__":
    main()
