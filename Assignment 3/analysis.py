# Andrew Nowotarski
# anowotarski3
# CS 7641 ML Spring 2020
# Assignment 3: Unsupervised Learning and Dimensionality Reduction

import pandas as pd
import numpy as np
import time
import plotting
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, homogeneity_completeness_v_measure
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import kurtosis, entropy
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import ShuffleSplit

def get_kmeans_clustering_predictions(features, cluster_size):

        # Train the clustering algo. #
        kmeans_model = KMeans(n_clusters= cluster_size, max_iter= 100)
        kmeans_model.fit(features)

        # Get the predictions. #
        kmeans_predictions = kmeans_model.predict(features)

        return kmeans_model, kmeans_predictions

def get_expectations_maximization_predictions(features, cluster_size):

        # Train the clustering algo. #
        em_model = BayesianGaussianMixture(n_components= cluster_size, reg_covar= 1e-0)
        em_model.fit(features)

        # Get the predictions. #
        em_predictions = em_model.predict(features)

        return em_model, em_predictions

def run_clustering_algorithms(features, labels, dataset_name, img_title):

    cluster_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

    cluster_baseline_results = pd.DataFrame(columns=['size', 'km_score', 'km_mutual_info', 'km_rand', 'km_homogeneity', 'km_completeness', 'km_v_measure', 'km_time',
                                            'em_score', 'em_mutual_info', 'em_rand', 'em_homgeneity', 'em_completeness', 'em_v_measure', 'em_time'])
    for n in cluster_sizes:

        ######################
        # KMeans Clustering. #
        ######################

        # Time the training / testing for Kmeans. #
        kmeans_start_time = time.time()

        # Train the clustering algo and get the predictions. #
        kmeans_model, kmeans_predictions = get_kmeans_clustering_predictions(features, n)

        print(kmeans_predictions)
        # Time the training / testing for Kmeans. #
        kmeans_end_time = time.time()

        # Get the metrics. #
        kmeans_homogeneity, kmeans_completeness, kmeans_v_measure = homogeneity_completeness_v_measure(labels.ravel(), kmeans_predictions)

        ##############################
        # Expectations Maximization. #
        ##############################

        # Time the training / testing to Expectations Maximization. #
        em_start_time = time.time()

        # Train the clustering algo. #
        em_model = BayesianGaussianMixture(n_components= n, reg_covar= 1e-0)
        em_model.fit(features)

        # Get the predictions. #
        em_predictions = em_model.predict(features)

        # Time the training / testing for Expectations Maximization. #
        em_end_time = time.time()

        # Get the metrics. 
        em_homogeneity, em_completeness, em_v_measure = homogeneity_completeness_v_measure(labels.ravel(), em_predictions)

        # Record the metrics for the algorithm / cluster. #
        cluster_baseline_results = cluster_baseline_results.append({
            'size': n,
            'km_score': -kmeans_model.score(features),
            'km_mutual_info': adjusted_mutual_info_score(labels.ravel(), kmeans_predictions),
            'km_rand': adjusted_rand_score(labels.ravel(), kmeans_predictions), 
            'km_homogeneity': kmeans_homogeneity,
            'km_completeness': kmeans_completeness,
            'km_v_measure': kmeans_v_measure,
            'km_time': kmeans_end_time - kmeans_start_time,
            'em_score': em_model.score(features), 
            'em_mutual_info': adjusted_mutual_info_score(labels.ravel(), em_predictions),
            'em_rand': adjusted_rand_score(labels.ravel(), em_predictions),
            'em_homgeneity': em_homogeneity,
            'em_completeness': em_completeness,
            'em_v_measure': em_v_measure,
            'em_time': em_end_time - em_start_time
        }, ignore_index=True)

    plotting.plot_cluster_scores(dataset_name, dataset_name + '/cluster_' + img_title + '.png', cluster_baseline_results)
    plotting.plot_cluster_cost(dataset_name, dataset_name + '/kmeans_cost_' + img_title + '.png', 'km', 'Sum of Squared Distances', cluster_baseline_results)
    plotting.plot_cluster_cost(dataset_name, dataset_name + '/expectation_maximization_cost_' + img_title + '.png', 'em', 'Likelihood', cluster_baseline_results)
    plotting.plot_cluster_speed(dataset_name, dataset_name + '/cluster_times_' + img_title + '.png', cluster_baseline_results)

def run_analysis(estimator, features, labels, dataset, modelType, analysisRound, validationXLabel, param, param_values):

    X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.20, random_state = 41)

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