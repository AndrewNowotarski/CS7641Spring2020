# Andrew Nowotarski
# anowotarski3
# CS 7641 ML Spring 2020
# Assignment 3: Unsupervised Learning and Dimensionality Reduction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import ShuffleSplit
from scipy.stats import kurtosis,entropy
from sklearn.model_selection import train_test_split

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
    plt.savefig('Charts/' + dataset + '/' + modelType + '/' + analysisRound + '/Learning Curve ' + modelType + '.png')
    plt.clf()

    # Plot n_samples vs fit_times
    plt.plot(train_sizes, fit_times_mean, 'o-')
    plt.fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    plt.xlabel("Training examples")
    plt.ylabel("Fit Times")
    plt.title("Scalability of the model " + modelType)
    plt.savefig('Charts/' + dataset + '/' + modelType + '/' + analysisRound + '/Samples Versus Fit Time ' + modelType + '.png')
    plt.clf()

    # Plot fit_time vs score
    plt.plot(fit_times_mean, test_scores_mean, 'o-')
    plt.fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    plt.xlabel("Fit Times")
    plt.ylabel("Score")
    plt.title("Performance of the model " + modelType)
    plt.savefig('Charts/' + dataset + '/' + modelType + '/' + analysisRound + '/Fit Time Versus Score ' + modelType + '.png')
    plt.clf()

# CODE SOURCED FROM
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html#sphx-glr-auto-examples-model-selection-plot-validation-curve-py
# https://chrisalbon.com/machine_learning/model_evaluation/plot_the_validation_curve/
def plot_validation_curve(estimator, X, y, dataset, modelType, analysisRound, xlabel, param_name, param_range):

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
    plt.plot(param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw)

    # Plot accuracy bands for training and test sets.
    plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2, color="darkorange", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2, color="navy", lw=lw)

    plt.title("Validation Curve " + modelType)
    plt.xlabel(xlabel)
    plt.ylabel("Accuracy Score")
    plt.legend(loc="best")

    plt.savefig('Charts/' + dataset + '/' + modelType + '/' + analysisRound + '/Validation Curve ' + modelType + '.png')
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

def plot_cluster_scores(title, img_title, cluster_results):

    x = cluster_results.iloc[:,0]

    # Plot the KMeans / EM Scores for various metrics. #
    plt.figure(figsize = (8,6))

    # Mutual Information. # 
    plt.plot(x, cluster_results["km_mutual_info"], color='blue', linewidth=4,label = "KM Mutual info")
    plt.plot(x, cluster_results["em_mutual_info"],linestyle='dashed',color='blue', linewidth=4,label = "EM Mutual info")

    # Random Index. #
    plt.plot(x, cluster_results["km_rand"],color='green', linewidth=4,label = "KM Rand Index")
    plt.plot(x, cluster_results["em_rand"],linestyle='dashed',color='green', linewidth=4,label = "EM Rand Index")

    # Homogeneity. #
    plt.plot(x, cluster_results["km_homogeneity"],color='red', linewidth=4,label = "KM Homogeneity")
    plt.plot(x, cluster_results["em_homgeneity"],linestyle='dashed',color='red', linewidth=4,label = "EM Homogeneity")

    # V-Measure. #
    plt.plot(x, cluster_results["km_v_measure"],color='orange', linewidth=4,label = "KM V-Measure");
    plt.plot(x, cluster_results["em_v_measure"],linestyle='dashed',color='orange', linewidth=4,label = "EM V-Measure")

    plt.title("KMeans / Expectation Maximization - " + title)
    plt.xlabel("Clusters")
    plt.ylabel("Score")
    plt.legend(prop={'size':13}, loc='lower right')

    plt.savefig('Charts/' + img_title)
    plt.clf()

def plot_cluster_cost(title, img_title, cluster_algo, ylabel, cluster_results):

    x = cluster_results.iloc[:,0]

    plt.figure(figsize = (8,6)) 

    plt.plot(x, cluster_results[cluster_algo + "_score"], marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4,label = "Clusters")

    plt.title(cluster_algo.upper() + " Cost - " + title)
    plt.xlabel("Clusters")
    plt.ylabel(ylabel)

    plt.savefig('Charts/' + img_title)
    plt.clf()

def plot_cluster_speed(title, img_title, cluster_results):

    # Plot KMeans / Expectation Maximization Times. #
    x = cluster_results.iloc[:,0]

    plt.figure(figsize = (8,6))
    plt.plot(x ,cluster_results["km_time"], label = "KMeans")
    plt.plot(x, cluster_results["em_time"], label = "Expectation Maximization")
    plt.title(title)
    plt.xlabel("Clusters")
    plt.ylabel("Time")
    plt.legend(loc='upper right')

    plt.savefig('Charts/' + img_title)
    plt.clf()

def plot_pca_eigen_values(title, img_title, fittedModel, features):

    # Get the explained variance and ratio. #
    explained_variance_ratio = pd.Series(fittedModel.explained_variance_ratio_)
    explained_variance = pd.Series(fittedModel.explained_variance_)

    # Plot the eigen values. #
    plt.figure(figsize= (8,6))

    explained_variance_ratio.plot(ylim = (0.,0.35),c = 'r',label = 'Explained Variance')
    ax = explained_variance.plot(kind = 'bar',ylim = (0.,0.35),label = "Explained Variance Ratio")

    ticks = ax.xaxis.get_ticklocs()
    ticklabels = [l.get_text() for l in ax.xaxis.get_ticklabels()]
    ax.xaxis.set_ticks(ticks[::10])
    ax.xaxis.set_ticklabels(ticklabels[::10]);
    plt.title("PCA Eigen Values Distribution - " + title)
    plt.xlabel("Features By Variance Ordered")
    plt.ylabel("Explained Variance / Ratio")
    plt.legend(loc='upper right')

    plt.savefig('Charts/' + img_title)
    plt.clf()

    print("Reduced Dimension: " + str(features.shape[1]-len([i for i in explained_variance_ratio if i >= 0.005])) + " out of " + str(features.shape[1]))
    print("Variance captured: " + str(sum([i for i in explained_variance_ratio if i >= 0.005])*100.) +  "%")

def plot_ica_kurtosis(title, img_title, fittedModel, ica_features, features):

    # Calculate Kurtosis for Features .#
    order = [-abs(kurtosis(ica_features[:,i])) for i in range(ica_features.shape[1])]
    ica_features = ica_features[:,np.array(order).argsort()]
    ica_res =  pd.Series([abs(kurtosis(ica_features[:,i])) for i in range(ica_features.shape[1])]);

    plt.figure(figsize=(8,6))
    ax = ica_res.plot(kind = 'bar',logy = True);
    ticks = ax.xaxis.get_ticklocs()
    ticklabels = [l.get_text() for l in ax.xaxis.get_ticklabels()]
    ax.xaxis.set_ticks(ticks[::10])
    ax.xaxis.set_ticklabels(ticklabels[::10]);

    plt.title("ICA Kurtosis Distribution - " + title)
    plt.xlabel("Features By Absolute Kurtosis Value Ordered")
    plt.ylabel("Absolute Kurtosis")

    plt.savefig('Charts/' + img_title)
    plt.clf()

    print("Reduced Dimension: " + str(features.shape[1]-len([i for i in ica_res if i >= 1.])) +  " out of " + str(features.shape[1]))

def plot_random_projections(title, img_title, fittedModel, rp_features, features):

    # Calculate some metric.. #
    print("Reduced Dimension: " + str(rp_features.shape[1]) +  " out of " + str(features.shape[1]))

def plot_svd(title, img_title, fittedModel, svd_features, features):

    print("Reduced Dimension: " + str(svd_features.shape[1]) +  " out of " + str(features.shape[1]))

def plot_roc_curves_dimensionality_reduction(classifiers, title, img_title):

    print(img_title)

    # Define a result table as a DataFrame.
    result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])

    # Train the models and record the results.
    for cls in classifiers:

        print("Working on " + cls.name)

        # Split the data into train / test sets.
        X_train, X_test, Y_train, Y_test = train_test_split(cls.features, cls.labels, test_size=0.20, random_state = 41)
        
        Y_train = Y_train.ravel()
        Y_test = Y_test.ravel()

        model = cls.model.fit(X_train, Y_train)
        yproba = model.predict_proba(X_test)[::,1]
        
        fpr, tpr, _ = roc_curve(Y_test.ravel(),  yproba)
        auc = roc_auc_score(Y_test.ravel(), yproba)
        
        result_table = result_table.append({'classifiers':cls.name,
                                            'fpr':fpr, 
                                            'tpr':tpr, 
                                            'auc':auc}, ignore_index=True)

    # Set name of the classifiers as index labels
    result_table.set_index('classifiers', inplace=True)

    # Plot the chart.
    print("Plotting the data...")

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

    plt.title(title, fontweight='bold', fontsize=15)
    plt.legend(prop={'size':13}, loc='lower right')

    plt.savefig(img_title)
    plt.clf()