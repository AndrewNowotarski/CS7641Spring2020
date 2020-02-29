# Andrew Nowotarski
# anowotarski3
# CS 7641 ML Spring 2020
# Assignment 2: Randomized Optimization

# Packages for matrix manipulation. 
import mlrose_hiive as mlrose
import numpy as np

# Packages for metrics and plotting.
import plotting

# Instantiate a Queens problem.
fitness = mlrose.Queens()
problem_fit = mlrose.DiscreteOpt(length = 15, fitness_fn= fitness)

# Solve the problem with genetic algorithm
plotting.plot_optimization_problem_fitness(problem_fit, 100, 2, 'N-Queens')

# CODE SOURCED FROM
# https://mlrose.readthedocs.io/en/stable/source/tutorial1.html

# Define alternative N-Queens fitness function for maximization problem.
def queens_max(state):

    # Initialize counter
    fitness_cnt = 0

    # For all pairs of queens
    for i in range(len(state) - 1):
        for j in range(i + 1, len(state)):

            # Check for horizontal, diagonal-up and diagonal-down attacks
            if (state[j] != state[i]) \
                and (state[j] != state[i] + (j - i)) \
                and (state[j] != state[i] - (j - i)):

                # If no attacks, then increment counter
                fitness_cnt += 1

    return fitness_cnt

# Initialize custom fitness function object.
cust_fitness = mlrose.CustomFitness(queens_max)