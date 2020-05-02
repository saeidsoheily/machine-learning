__author__ = 'Saeid SOHILY-KHAH'
"""
Machine learning algorithms: Genetic Algorithm for Solving Mathematical Equality Problem
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import random, randint


# Generate a member of the population (i.e. chromosome)
def generate_individual(length):
    '''
    Generate a member of the population
    :param length: number of genes per individual
    :return: individual (i.e. chromosome)
    '''
    individual = np.random.randint(2, size=length)
    return individual


# Generate a population
def generate_population(count, length):
    '''
    Generate a population (i.e. number of individuals)
    :param count: number of individuals in the population
    :param length: number of genes per individual
    :return:
    '''
    population = np.array([generate_individual(length) for _ in range(count)])
    return population


# Convert an individual to a float number
def convert_individual_to_number(individual):
    '''
    Convert an individual (i.e. chromosome) to a float number
    :param individual:
    :return: number: a float number
    '''
    integer_part = sum([individual[i] * (2 ** i) for i in range(0, len(individual)//2)])
    decimal_part = sum([individual[i] * (2 ** i) for i in range(len(individual)//2, len(individual))])
    number = integer_part + (decimal_part/(10**len(str(decimal_part))))
    return number


# Define the individual's fitness
def fitness(individual, goal_value):
    '''
    Define the individual's fitness
    :param individual: chromosome
    :param goal_value:
    :return: fitness error
    '''
    epsilon = 1e-10 # to avoid division by zero
    x = convert_individual_to_number(individual) # convert the chromosome to a float number
    y = x * - np.log(1 / (x + epsilon)) # pre-defined equality function
    return abs(goal_value - y)


# Compute average fitness of the population
def population_fitness(population, goal_value):
    '''
    Compute average fitness of the population
    :param population:
    :param goal_value:
    :return: average_fitness
    '''
    sum_fitness = np.sum(np.array([fitness(individual, goal_value) for individual in population])) * 1.0
    average_fitness = sum_fitness / len(population)
    return average_fitness


# Evolutionary algorithm
def evolution(population, goal_value, retain=0.2, random_select=0.05, mutate=0.01):
    '''
    Evolutionary algorithm
    :param population:
    :param goal_value:
    :param retain: survival
    :param random_select: retain + an additional of other individuals
    :param mutate:
    :return: parents as new population
    '''
    fitness_lst = [fitness(individual, goal_value) for individual in population] # fitness values of the population
    indices_sorted = list(np.argsort(np.array(fitness_lst))) # indices of sorted individuals' fitness
    population_sorted = [list(population[idx]) for idx in indices_sorted] # sorted population by individual's fitness

    # Keep only survival individuals
    retain_length = int(len(population_sorted) * retain)
    parents = population_sorted[:retain_length]

    # Randomly add other individuals to promote genetic diversity
    for individual in population_sorted[retain_length:]:
        if random_select > random():
            parents.append(individual)

    # Mutate some individuals
    for individual in parents:
        if mutate > random():
            position_to_mutate = randint(0, len(individual) - 1)
            individual[position_to_mutate] = abs(individual[position_to_mutate] - 1)

    # Crossover parents to create children
    parents_length = len(parents)
    desired_length = len(population) - parents_length
    children = []
    while len(children) < desired_length:
        male = randint(0, parents_length - 1)
        female = randint(0, parents_length - 1)
        if male != female:
            male = parents[male]
            female = parents[female]
            half = int(len(male) / 2)
            child = [1] * len(parents[0])
            child[:half] = male[:half]
            child[half:] = female[half:]
            children.append(child)
    parents.extend(children)
    return parents


# Genetic algorithm
def ga(length, goal_value, population_size, iter_number):
    '''
    Genetic algorithm
    :param length: number of genes per each chromosome (i.e. individuals' length)
    :param goal_value:
    :param population_size: total number of chromosomes (i.e. size of the population)
    :param iter_number:
    :return:
    '''
    population = generate_population(population_size, length)
    population_fitness_history = [population_fitness(population, goal_value), ]
    for i in range(iter_number):
        population = evolution(population, goal_value)
        population_fitness_history.append(population_fitness(population, goal_value))
    return population, population_fitness_history


# ------------------------------------------------ MAIN ----------------------------------------------------
if __name__ == '__main__':
    # Pre-defined example equality function (and its solution)
    solution = 2.5
    y = solution * - np.log(1 / solution) # goal_value

    # Genetic algorithm
    chromosome_length = 10  # the number of genes per individual
    population_total_size = 100
    iteration_number = 20
    results_vector, fitness_history = ga(chromosome_length, y, population_total_size, iteration_number)
    ga_solution_estimation = convert_individual_to_number(results_vector[0])  # top 1 of the population

    # Plot settings
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))  # set figure shape and size
    fig.tight_layout(pad=3.0)  # set the spacing between subplots in Matplotlib

    # Plot results
    axes.set_xlabel('Iteration')
    axes.set_ylabel('Average Population Error')
    axes.set_title('GENETIC ALGORITHM: AVERAGE POPULATION ERROR', fontsize=12)
    axes.plot(fitness_history)
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False, edgecolor='none', linewidth=0)
    results = (r'SOLUTION={:.2f}' + '\n' + r'GA ESTIMATION={:.2f}').format(solution, ga_solution_estimation)
    axes.legend([extra], [results])

    # To save the plot locally
    plt.savefig('genetic_algorithm.png', bbox_inches='tight')
    plt.show()