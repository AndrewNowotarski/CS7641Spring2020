# Andrew Nowotarski
# anowotarski3
# CS 7641 ML Spring 2020
# Assignment 1: Supervised Learning

# Packages for matrix manipulation. 
import numpy as np
import pandas as pd

# Packages for plotting.
import plotting

# Packages for metrics.
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

def run_analysis(estimator, X_train, Y_train, X_test, Y_test, dataset, modelType, analysisRound, validationXLabel, param, param_values):

    # Train.
    estimator.fit(X_train, Y_train.ravel())

    # Test.
    predictions = estimator.predict(X_test)

    # Confusion matrix.
    tn, fp, fn, tp = metrics.confusion_matrix(Y_test.ravel(), predictions).ravel()

    print(metrics.confusion_matrix(Y_test.ravel(), predictions))
    print("Accuracy: " + str(metrics.accuracy_score(Y_test.ravel(), predictions)))
    print("Precision: " + str(metrics.precision_score(Y_test.ravel(), predictions)))
    print("Recall: " + str(metrics.recall_score(Y_test.ravel(), predictions)))

    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.

    # Learning Curve Analysis.
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=10)
    plotting.plot_learning_curve(estimator, X_train, Y_train.ravel(), dataset, modelType, analysisRound, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

    # Model Complexity / Validation Curve Analysis.
    plotting.plot_validation_curve(estimator, X_train, Y_train.ravel(), dataset, modelType, analysisRound, validationXLabel, param, param_values)