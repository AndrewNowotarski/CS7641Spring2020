# Andrew Nowotarski
# anowotarski3
# CS 7641 ML Spring 2020
# Assignment 1: Supervised Learning

# Packages for matrix manipulation. 
import numpy as np
import pandas as pd
import analysis

# Packages for metrics and plotting.
import plotting

# Packages for 5 algorithms.
from sklearn import svm
from sklearn import tree
from sklearn import neighbors
from sklearn import ensemble
from sklearn import neural_network
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit

# Sourced from https://www.kaggle.com/ak1352/titanic-cl
titanic = pd.read_csv("Titanic/titanic_survivor.csv")

x = titanic.iloc[:,:-1]
y = np.array(titanic.iloc[:,-1:])

# Split into training and test sets (80-20).
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.20, random_state = 41)

##########################
# Run KNN Analysis. #
##########################

# Round 1. - Number of neighbors
# analysis.run_analysis(neighbors.KNeighborsClassifier(), X_train, Y_train, X_test, Y_test, 'Titanic', 'KNN', 'Round_1', 'Number of Neighbors', 'n_neighbors', [1, 10, 25, 50, 75, 100])

# Round 2 - Distance Measure
# analysis.run_analysis(neighbors.KNeighborsClassifier(n_neighbors=25), X_train, Y_train, X_test, Y_test, 'Titanic', 'KNN', 'Round_2', 'Distance Measure', 'p', [1, 2])

# Round 3. - Algorithm
# analysis.run_analysis(neighbors.KNeighborsClassifier(n_neighbors=25, p = 1), X_train, Y_train, X_test, Y_test, 'Titanic', 'KNN', 'Round_3', 'Algorithm', 'algorithm', ['ball_tree', 'kd_tree','brute'])

###############################
# Run Decision Tree Analysis. #
###############################

# Round 1. - Criterion
#analysis.run_analysis(tree.DecisionTreeClassifier(), X_train, Y_train, X_test, Y_test, 'Titanic', 'Decision Tree', 'Round_1', 'Criterion', 'criterion', ['gini', 'entropy'])

# Round 2. - Max Leaf Nodes
#analysis.run_analysis(tree.DecisionTreeClassifier(criterion='gini'), X_train, Y_train, X_test, Y_test, 'Titanic', 'Decision Tree', 'Round_2', 'Max Leaf Nodes', 'max_leaf_nodes', [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100])

# Round 3. - Max Depth
#analysis.run_analysis(tree.DecisionTreeClassifier(criterion='gini', max_leaf_nodes=5), X_train, Y_train, X_test, Y_test, 'Titanic', 'Decision Tree', 'Round_3', 'Max Depth', 'max_depth', [1, 10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100])

#########################################################
# Run Boosting Analysis - Adaboost with decision trees. #
#########################################################

# Round 1. - Number of estimators.
#analysis.run_analysis(ensemble.AdaBoostClassifier(), X_train, Y_train, X_test, Y_test, 'Titanic', 'AdaBoost', 'Round_1', 'Number of trees', 'n_estimators', [2, 5, 10, 25, 50, 75, 100])

# Round 2. - Learning Rate
#analysis.run_analysis(ensemble.AdaBoostClassifier(n_estimators=10), X_train, Y_train, X_test, Y_test, 'Titanic', 'AdaBoost', 'Round_2', 'Learning Rate', 'learning_rate', [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])

# Round 3. - Algorithm.
#analysis.run_analysis(ensemble.AdaBoostClassifier(n_estimators=10, learning_rate= 0.3), X_train, Y_train, X_test, Y_test, 'Titanic', 'AdaBoost', 'Round_3', 'Algorithm', 'algorithm', ['SAMME', 'SAMME.R'])

########################################
# Run Support Vector Machine Analysis. #
########################################

# Round 1. - Kernal
#analysis.run_analysis(svm.SVC(), X_train, Y_train, X_test, Y_test, 'Titanic', 'SVM', 'Round_1', 'Kernel', 'kernel', ['linear', 'poly', 'rbf', 'sigmoid'])

# Round 2. - Degree
#analysis.run_analysis(svm.SVC(kernel= 'linear'), X_train, Y_train, X_test, Y_test, 'Titanic', 'SVM', 'Round_2', 'Degree', 'degree', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Round 3. - Max Iterations
#analysis.run_analysis(svm.SVC(kernel= 'linear', degree= 10), X_train, Y_train, X_test, Y_test, 'Titanic', 'SVM', 'Round_3', 'Max Iterations', 'max_iter', [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

################################
# Run Neural Network Analysis. #
################################

# Round 1. - Activation
#analysis.run_analysis(neural_network.MLPClassifier(), X_train, Y_train, X_test, Y_test, 'Titanic', 'Neural Network', 'Round_1', 'Activation', 'activation', ['identity', 'logistic', 'tanh', 'relu'])

# Round 2. - Solver
#analysis.run_analysis(neural_network.MLPClassifier(activation='identity'), X_train, Y_train, X_test, Y_test, 'Titanic', 'Neural Network', 'Round_2', 'Solver', 'solver', ['lbfgs', 'sgd', 'adam'])

# Round 3. - Hidden layer size
#analysis.run_analysis(neural_network.MLPClassifier(activation='identity', solver='adam'), X_train, Y_train, X_test, Y_test, 'Titanic', 'Neural Network', 'Round_3', 'Beta_1', 'beta_1', [.1, .2, .3, .4, .5, .6, .7, .8, .9])

##################################################
# Final Analysis - Compare Base To Tuned Models. #
##################################################

# Use default classifier settings. #
classifiers = [tree.DecisionTreeClassifier(),
                neighbors.KNeighborsClassifier(),
                svm.SVC(probability=True),
                neural_network.MLPClassifier(),
                ensemble.AdaBoostClassifier()]

plotting.plot_roc_curves(classifiers, X_train, Y_train.ravel(), X_test, Y_test.ravel(), 'Titanic', 'BaseLine_ROC_Curve.png')

# Tuned classifier settings. #
tunedClassifiers = [tree.DecisionTreeClassifier(criterion='gini', max_leaf_nodes=5, max_depth=10),
                neighbors.KNeighborsClassifier(n_neighbors=25, p = 1, algorithm='kd_tree'),
                svm.SVC(probability=True, kernel= 'linear', degree=10, max_iter=-1),
                neural_network.MLPClassifier(activation='identity', solver='adam', beta_1=0.4),
                ensemble.AdaBoostClassifier(n_estimators=10, learning_rate= 0.3, algorithm='SAMME.R')]

plotting.plot_roc_curves(tunedClassifiers, X_train, Y_train.ravel(), X_test, Y_test.ravel(), 'Titanic', 'Tuned_ROC_Curve.png')