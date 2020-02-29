# Andrew Nowotarski
# anowotarski3
# CS 7641 ML Spring 2020
# Assignment 2: Randomized Optimization

import pandas as pd
import numpy as np
import mlrose_hiive as mlrose
import matplotlib.pyplot as plt
import timeit
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics

# CODE SOURCED FROM:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
def plot_learning_curve(estimator, X, y, dataset, modelType, analysisRound, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    plt.legend(loc="best")
    plt.xlabel("Training Size")
    plt.ylabel("Score")
    plt.title("Learning Curve " + modelType)
    plt.savefig('Charts/' + dataset + '/' + analysisRound + '/' + modelType + '/Learning Curve ' + modelType + '.png')
    plt.clf()

    # Plot n_samples vs fit_times
    plt.plot(train_sizes, fit_times_mean, 'o-')
    plt.fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    plt.xlabel("Training examples")
    plt.ylabel("Fit Times")
    plt.title("Scalability of the model " + modelType)
    plt.savefig('Charts/' + dataset + '/' + analysisRound + '/' + modelType + '/Samples Versus Fit Time ' + modelType + '.png')
    plt.clf()

    # Plot fit_time vs score
    plt.plot(fit_times_mean, test_scores_mean, 'o-')
    plt.fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    plt.xlabel("Fit Times")
    plt.ylabel("Score")
    plt.title("Performance of the model " + modelType)
    plt.savefig('Charts/' + dataset + '/' + analysisRound + '/' + modelType + '/Fit Time Versus Score ' + modelType + '.png')
    plt.clf()

# CODE SOURCED FROM
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html#sphx-glr-auto-examples-model-selection-plot-validation-curve-py
# https://chrisalbon.com/machine_learning/model_evaluation/plot_the_validation_curve/
def plot_validation_curve(estimator, X, y, dataset, modelType, analysisRound, xlabel, param_name, param_range, x_labels = []):

    # Calculate accuracy on training and test set using range of parameter values.
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        scoring="accuracy", n_jobs=1)

    # Calculate mean and standard deviation for training set scores.
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)

    # Calculate mean and standard deviation for test set scores.
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot mean accuracy scores for training and test sets.
    lw = 2
    if (len(x_labels) > 0):
        plt.plot(x_labels, train_scores_mean, label="Training score", color="darkorange", lw=lw)
        plt.plot(x_labels, test_scores_mean, label="Cross-validation score", color="navy", lw=lw)

        # Plot accuracy bands for training and test sets.
        plt.fill_between(x_labels, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2, color="darkorange", lw=lw)
        plt.fill_between(x_labels, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2, color="navy", lw=lw)

    else:
        plt.plot(param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw)
        plt.plot(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw)

        # Plot accuracy bands for training and test sets.
        plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2, color="darkorange", lw=lw)
        plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2, color="navy", lw=lw)

 
    plt.title("Validation Curve " + modelType)
    plt.xlabel(xlabel)
    plt.ylabel("Accuracy Score")
    plt.legend(loc="best")

    plt.savefig('Charts/' + dataset + '/' + analysisRound + '/' +  modelType + '/Validation Curve ' + modelType + '.png')
    plt.clf()

# CODE SOURCED FROM
# https://abdalimran.github.io/2019-06-01/Drawing-multiple-ROC-Curves-in-a-single-plot
def plot_roc_curves(classifiers, X_train, Y_train, X_test, Y_test, dataset, filename):

    # Define a result table as a DataFrame.
    result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])

    # Train the models and record the results.
    for cls in classifiers:
 
        model = cls.fit(X_train, Y_train)

        yproba = model.predict_proba(X_test)[::,1]
        
        fpr, tpr, _ = roc_curve(Y_test,  yproba)
        auc = roc_auc_score(Y_test, yproba)
        
        result_table = result_table.append({'classifiers':cls.__class__.__name__,
                                            'fpr':fpr, 
                                            'tpr':tpr, 
                                            'auc':auc}, ignore_index=True)

    # Set name of the classifiers as index labels
    result_table.set_index('classifiers', inplace=True)

    # Plot the chart.
    fig = plt.figure(figsize=(8,6))

    for i in result_table.index:
        plt.plot(result_table.loc[i]['fpr'], 
                result_table.loc[i]['tpr'], 
                label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))
        
    plt.plot([0,1], [0,1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("False Positive Rate", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)

    plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
    plt.legend(prop={'size':13}, loc='lower right')

    plt.savefig('Charts/' + dataset + '/' + filename)
    plt.clf()

# Output result of confusion matrix.
def output_statistics(model, x_train, y_train, x_test, y_test, name):
    print("***********************")
    print(name)
    print("***********************")

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    tn, fp, fn, tp = metrics.confusion_matrix(y_test.ravel(), predictions).ravel()

    print("TN: " + str(tn))
    print("FP: " + str(fp))
    print("FN: " + str(fn))
    print("TP: " + str(tp))
    print("Accuracy: " + str(metrics.accuracy_score(y_test.ravel(), predictions)))
    print("Precision: " + str(metrics.precision_score(y_test.ravel(), predictions)))
    print("Recall: " + str(metrics.recall_score(y_test.ravel(), predictions)))

def plot_optimization_problem_fitness(fitness_function, iterations, random_state, title):

    start_rhc = timeit.default_timer()
    rhc_best_state, rhc_best_fitness, rch_fitness_curve =  mlrose.random_hill_climb(fitness_function, max_iters=iterations, random_state= random_state, restarts=10, curve=True)
    rhc_elapsed = timeit.default_timer() - start_rhc

    start_sa = timeit.default_timer()
    sa_best_state, sa_best_fitness, sa_fitness_curve =  mlrose.simulated_annealing(fitness_function, max_iters=iterations, random_state= random_state, curve=True)
    sa_elapsed = timeit.default_timer() - start_sa

    start_ga = timeit.default_timer()
    ga_best_state, ga_best_fitness, ga_fitness_curve =  mlrose.genetic_alg(fitness_function, max_iters=iterations, random_state= random_state, curve=True)
    ga_elapsed = timeit.default_timer() - start_ga

    start_mimic = timeit.default_timer()
    mimic_best_state, mimic_best_fitness, mimic_fitness_curve =  mlrose.mimic(fitness_function, max_iters=iterations, random_state= random_state, curve=True)
    mimic_elapsed = timeit.default_timer() - start_mimic

    # Fill in arrays.
    rch_fitness_curve_bf = np.full(iterations, rhc_best_fitness)
    rch_fitness_curve_bf[:rch_fitness_curve.shape[0]] = rch_fitness_curve

    sa_fitness_curve_bf = np.full(iterations, sa_best_fitness)
    sa_fitness_curve_bf[:sa_fitness_curve.shape[0]] = sa_fitness_curve

    ga_fitness_curve_bf = np.full(iterations, ga_best_fitness)
    ga_fitness_curve_bf[:ga_fitness_curve.shape[0]] = ga_fitness_curve

    mimic_fitness_curve_bf = np.full(iterations, mimic_best_fitness)
    mimic_fitness_curve_bf[:mimic_fitness_curve.shape[0]] = mimic_fitness_curve

    # Plot the convergance times.
    plot_ro_algo_times(rhc_elapsed, ga_elapsed, sa_elapsed, mimic_elapsed, title)

    # Plot the fitness over iterations.
    fig = plt.figure(figsize=(8,6))

    plt.plot(rch_fitness_curve_bf, label="RHC")
    plt.plot(sa_fitness_curve_bf, label="SA")
    plt.plot(ga_fitness_curve_bf, label="GA")
    plt.plot(mimic_fitness_curve_bf, label="MIMIC")

    plt.xlabel("Number of Iterations")
    plt.xticks(np.arange(0.0, iterations, step=iterations / 10))

    plt.ylabel("Fitness Function")

    plt.title(title)
    plt.legend(prop={'size':13}, loc='lower right')

    plt.savefig('Charts/OptimizationProblems/' + title + '.png')
    plt.clf()

# CODE SOURCED FROM
# https://pythonspot.com/matplotlib-bar-chart/
def plot_ro_algo_times(rch_time, ga_time, sa_time, mimic_time, title):

    fig = plt.figure(figsize=(8,6))

    algos = ("RHC", "GA", "SA", "MIMIC")
    y_pos = np.arange(len(algos))
    performance = [rch_time, ga_time, sa_time, mimic_time]

    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, algos)
    plt.ylabel('Time To Converge')
    plt.xlabel('Algorithm')
    plt.title(title)

    plt.savefig('Charts/OptimizationProblems/' + title + ' - TimeComplexity.png')
    plt.clf()