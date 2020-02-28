# Andrew Nowotarski
# anowotarski3
# CS 7641 ML Spring 2020
# Assignment 2: Randomized Optimization

# Packages for matrix manipulation. 
import numpy as np
import pandas as pd
import analysis

# Packages for metrics and plotting.
import plotting
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import accuracy_score

# Packages for algorithms.
from sklearn import neural_network
import mlrose_hiive as ml
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit

# Sourced from https://www.kaggle.com/ak1352/titanic-cl
titanic = pd.read_csv("Titanic/titanic_survivor.csv")

x = titanic.iloc[:,:-1]
y = np.array(titanic.iloc[:,-1:])

# Split into training and test sets (80-20).
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.20, random_state = 41)

###################################
# Run Random Hill Climb Analysis. #
###################################

# Tune number of restarts. #
# analysis.run_analysis(ml.NeuralNetwork(hidden_nodes=[100], activation = 'identity', algorithm = 'random_hill_climb', learning_rate= 0.001, max_iters=200, is_classifier= True),
#     X_train, Y_train, X_test, Y_test, 'Titanic', 'RHC', 'Round_1', 'Restarts', 'restarts', [0, 1, 2, 5, 10, 15, 20, 25, 50]
# )

# Tune number of iterations. #
# analysis.run_analysis(ml.NeuralNetwork(hidden_nodes=[100], activation = 'identity', algorithm = 'random_hill_climb', learning_rate= 0.001, restarts= 15, is_classifier= True),
#     X_train, Y_train, X_test, Y_test, 'Titanic', 'RHC', 'Round_2', 'Iterations', 'max_iters', [0, 10, 100, 500, 1000, 2500, 5000]
# )

# Final Learning Curve. #
# analysis.run_analysis(ml.NeuralNetwork(hidden_nodes=[100], activation = 'identity', algorithm = 'random_hill_climb', learning_rate= 0.001, max_iters=1000, restarts= 15, is_classifier= True),
#     X_train, Y_train, X_test, Y_test, 'Titanic', 'RHC', 'Final', '', '', []
# )

####################################
# Run Simulated Annealing Analysis #
####################################

# Tune Schedule Algorithm. #
# analysis.run_analysis(ml.NeuralNetwork(hidden_nodes=[100], activation = 'identity', algorithm = 'simulated_annealing', learning_rate= 0.001, max_iters=200, is_classifier= True),
#     X_train, Y_train, X_test, Y_test, 'Titanic', 'SA', 'Round_1', 'Schedule', 'schedule', [ml.GeomDecay(), ml.ArithDecay(), ml.ExpDecay()], ['GeomDecay', 'ArithDecay', 'ExpDecay']
# )

# Tune number of iterations.
# analysis.run_analysis(ml.NeuralNetwork(hidden_nodes=[100], activation = 'identity', algorithm = 'simulated_annealing', learning_rate= 0.001, schedule= ml.GeomDecay(), is_classifier= True),
#     X_train, Y_train, X_test, Y_test, 'Titanic', 'SA', 'Round_2', 'Iterations', 'max_iters', [0, 10, 100, 500, 1000, 2500, 5000]
# )

# Final Learning Curve. #
# analysis.run_analysis(ml.NeuralNetwork(hidden_nodes=[100], activation = 'identity', algorithm = 'simulated_annealing', learning_rate= 0.001, schedule= ml.GeomDecay(), max_iters=2500, is_classifier= True),
#     X_train, Y_train, X_test, Y_test, 'Titanic', 'SA', 'Final', '', '', []
# )

##################################
# Run Genetic Algorithm Analysis #
##################################

# Tune population size. #
# analysis.run_analysis(ml.NeuralNetwork(hidden_nodes=[100], activation = 'identity', algorithm = 'genetic_alg', learning_rate= 0.001, max_iters=200, is_classifier= True),
#     X_train, Y_train, X_test, Y_test, 'Titanic', 'GA', 'Round_1', 'Population Size', 'pop_size', [25, 50, 75, 100, 150, 200, 250, 500, 750, 1000]
# )

# Tune mutation probability. #
# analysis.run_analysis(ml.NeuralNetwork(hidden_nodes=[100], activation = 'identity', algorithm = 'genetic_alg', learning_rate= 0.001, max_iters=200, pop_size= 150, is_classifier= True),
#     X_train, Y_train, X_test, Y_test, 'Titanic', 'GA', 'Round_2', 'Mutation Probability', 'mutation_prob', [.1, .2, .3, .4, .5, .6, .7, .8, .9]
# )

# Tune number of iterations. #
analysis.run_analysis(ml.NeuralNetwork(hidden_nodes=[100], activation = 'identity', algorithm = 'genetic_alg', learning_rate= 0.001, pop_size= 150, mutation_prob=0.5, is_classifier= True),
    X_train, Y_train, X_test, Y_test, 'Titanic', 'GA', 'Round_3', 'Iterations', 'max_iters', [0, 10, 100, 150, 200, 250, 500]
)

# Final Learning Curve. #
# analysis.run_analysis(ml.NeuralNetwork(hidden_nodes=[100], activation = 'identity', algorithm = 'genetic_alg', learning_rate= 0.001, pop_size= 150, mutation_prob=0.5, max_iters=100, is_classifier= True),
#     X_train, Y_train, X_test, Y_test, 'Titanic', 'GA', 'Final', '', '', []
# )

##############################
# Run Baseline Sklearn Model #
##############################

# Final Learning Curve. #
# analysis.run_analysis(neural_network.MLPClassifier(activation='identity', solver='adam', beta_1=0.4),
#     X_train, Y_train, X_test, Y_test, 'Titanic', 'Sklearn Baseline', 'Final', '', '', []
# )

##################################################
# Final Analysis - Compare Base To Tuned Models. #
##################################################

# Dump out the metrics for the baselines.
# plotting.output_statistics(neural_network.MLPClassifier(activation='identity', solver='adam', beta_1=0.4), X_train, Y_train, X_test, Y_test, "NN Baseline")

# plotting.output_statistics(ml.NeuralNetwork(hidden_nodes=[100], activation = 'identity', algorithm = 'random_hill_climb', learning_rate= 0.001, max_iters=200, is_classifier= True), 
#     X_train, Y_train, X_test, Y_test, "Random Hill Climb Baseline")

# plotting.output_statistics(ml.NeuralNetwork(hidden_nodes=[100], activation = 'identity', algorithm = 'simulated_annealing', learning_rate= 0.001, max_iters=200, is_classifier= True), 
#     X_train, Y_train, X_test, Y_test, "Simulated Annealing Baseline")

# plotting.output_statistics(ml.NeuralNetwork(hidden_nodes=[100], activation = 'identity', algorithm = 'genetic_alg', learning_rate= 0.001, max_iters=200, is_classifier= True),
#     X_train, Y_train, X_test, Y_test, "Genetic Algorithm Baseline")

# # Dump out the metrics for the tuned models.
# plotting.output_statistics(ml.NeuralNetwork(hidden_nodes=[100], activation = 'identity', algorithm = 'random_hill_climb', learning_rate= 0.001, max_iters=1000, restarts= 15, is_classifier= True), 
#     X_train, Y_train, X_test, Y_test, "Random Hill Climb Tuned")

# plotting.output_statistics(ml.NeuralNetwork(hidden_nodes=[100], activation = 'identity', algorithm = 'simulated_annealing', learning_rate= 0.001, schedule= ml.GeomDecay(), max_iters=2500, is_classifier= True), 
#     X_train, Y_train, X_test, Y_test, "Simulated Annealing Tuned")

# plotting.output_statistics(ml.NeuralNetwork(hidden_nodes=[100], activation = 'identity', algorithm = 'genetic_alg', learning_rate= 0.001, pop_size= 150, mutation_prob=0.5, max_iters=100, is_classifier= True),
#     X_train, Y_train, X_test, Y_test, "Genetic Algorithm Tuned")
