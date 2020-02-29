# Andrew Nowotarski
# anowotarski3
# CS 7641 ML Spring 2020
# Assignment 2: Randomized Optimization

# Packages for matrix manipulation. 
import mlrose_hiive as mlrose
import numpy as np

# Packages for metrics and plotting.
import plotting

fitness = mlrose.FourPeaks(t_pct=0.15)
problem_fit = mlrose.DiscreteOpt(length=35, fitness_fn=fitness)

# Solve the problem with genetic algorithm
plotting.plot_optimization_problem_fitness(problem_fit, 100, 2, 'Four Peaks')

