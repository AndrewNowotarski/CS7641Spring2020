# Andrew Nowotarski
# anowotarski3
# CS 7641 ML Spring 2020
# Assignment 2: Randomized Optimization

# Packages for matrix manipulation. 
import mlrose_hiive as mlrose
import numpy as numpy

# Packages for metrics and plotting.
import plotting

# Example code sourced from 
# https://mlrose.readthedocs.io/en/stable/source/tutorial2.html

# Create list of city coordinates.
coords_list = [(1, 1), (4, 2), (5, 2), (6, 4), (4, 4), (3, 6), (1, 5), (2, 3)]

# Initialize fitness function using coords_list.
fitness_coords = mlrose.TravellingSales(coords= coords_list)

# Define optimization problem object.
problem_fit = mlrose.TSPOpt(length = 8, fitness_fn= fitness_coords, maximize=False)

# Solve the problem with genetic algorithm
plotting.plot_optimization_problem_fitness(problem_fit, 100, 2, 'Traveling Sales Person')

