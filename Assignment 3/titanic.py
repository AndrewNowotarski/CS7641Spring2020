# Andrew Nowotarski
# anowotarski3
# CS 7641 ML Spring 2020
# Assignment 3: Unsupervised Learning and Dimensionality Reduction

import pandas as pd
import classifier as cs
import numpy as np
import time
import plotting
import analysis
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, homogeneity_completeness_v_measure
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import kurtosis, entropy
from sklearn import neural_network

# Sourced from https://www.kaggle.com/ak1352/titanic-cl
titanic = pd.read_csv("Titanic/titanic_survivor.csv")
dataset_name = 'Titanic'

features = titanic.iloc[:,:-1]
labels = np.array(titanic.iloc[:,-1:])

######################################
# Part 1. K-Means and EM Clustering. #
######################################

#analysis.run_clustering_algorithms(features, labels, dataset_name, 'baseline')

#####################################
# Part 2. Dimensionality Reduction. #
#####################################

########
# PCA. #
########
# pca = PCA()
# pca_features = pca.fit_transform(features)

# plotting.plot_pca_eigen_values(dataset_name, dataset_name + '/pca_baseline.png', pca, features)

########
# LCA. #
########
# ica = FastICA(n_components=20, max_iter=5000, tol=.15)
# ica_features = ica.fit_transform(features)

# plotting.plot_ica_kurtosis(dataset_name, dataset_name + '/ica_baseline.png', ica, ica_features, features)

###########################
# Randomized Projections. #
###########################
# rp = GaussianRandomProjection(n_components=20)
# rp_features = rp.fit_transform(features)

# plotting.plot_random_projections(dataset_name, dataset_name + '/rp_baseline.png', rp, rp_features, features)

##################
# Truncated SVD. #
##################
# svd = TruncatedSVD(n_components=20)
# svd_features = svd.fit_transform(features)

# plotting.plot_svd(dataset_name, dataset_name + '/svd_baseline.png', svd, svd_features, features)

###########################################################
# Part 3. Clustering with Dimensionally Reduced Features. #
###########################################################

# analysis.run_clustering_algorithms(pca_features, labels, dataset_name, 'pca_reduced')
# analysis.run_clustering_algorithms(ica_features, labels, dataset_name, 'ica_reduced')
# analysis.run_clustering_algorithms(rp_features, labels, dataset_name, 'randomized_projections_reduced')
# analysis.run_clustering_algorithms(svd_features, labels, dataset_name, 'svd_reduced')

####################################################
# Part 4. Run Neural Networks with Projected Data. #
####################################################

# Tune PCA. #
#analysis.run_analysis(neural_network.MLPClassifier(), pca_features, labels, 'Titanic', 'PCA', 'Round_1', 'Activation', 'activation', ['identity', 'logistic', 'tanh', 'relu'])
#analysis.run_analysis(neural_network.MLPClassifier(activation='identity'), pca_features, labels, 'Titanic', 'PCA', 'Round_2', 'Solver', 'solver', ['lbfgs', 'sgd', 'adam'])
#analysis.run_analysis(neural_network.MLPClassifier(activation='identity', solver='sgd'), pca_features, labels, 'Titanic', 'PCA', 'Round_3', 'Beta_1', 'beta_1', [.1, .2, .3, .4, .5, .6, .7, .8, .9])

# Tune ICA. #
#analysis.run_analysis(neural_network.MLPClassifier(), ica_features, labels, 'Titanic', 'ICA', 'Round_1', 'Activation', 'activation', ['identity', 'logistic', 'tanh', 'relu'])
#analysis.run_analysis(neural_network.MLPClassifier(activation='identity'), ica_features, labels, 'Titanic', 'ICA', 'Round_2', 'Solver', 'solver', ['lbfgs', 'sgd', 'adam'])
#analysis.run_analysis(neural_network.MLPClassifier(activation='identity', solver='adam'), ica_features, labels, 'Titanic', 'ICA', 'Round_3', 'Beta_1', 'beta_1', [.1, .2, .3, .4, .5, .6, .7, .8, .9])

# Tune Randomized Projections. #
#analysis.run_analysis(neural_network.MLPClassifier(), rp_features, labels, 'Titanic', 'RP', 'Round_1', 'Activation', 'activation', ['identity', 'logistic', 'tanh', 'relu'])
#analysis.run_analysis(neural_network.MLPClassifier(activation='identity'), rp_features, labels, 'Titanic', 'RP', 'Round_2', 'Solver', 'solver', ['lbfgs', 'sgd', 'adam'])
#analysis.run_analysis(neural_network.MLPClassifier(activation='identity', solver='sgd'), rp_features, labels, 'Titanic', 'RP', 'Round_3', 'Beta_1', 'beta_1', [.1, .2, .3, .4, .5, .6, .7, .8, .9])

# Tune SVD. #
#analysis.run_analysis(neural_network.MLPClassifier(), svd_features, labels, 'Titanic', 'SVD', 'Round_1', 'Activation', 'activation', ['identity', 'logistic', 'tanh', 'relu'])
#analysis.run_analysis(neural_network.MLPClassifier(activation='identity'), svd_features, labels, 'Titanic', 'SVD', 'Round_2', 'Solver', 'solver', ['lbfgs', 'sgd', 'adam'])
#analysis.run_analysis(neural_network.MLPClassifier(activation='identity', solver='adam'), svd_features, labels, 'Titanic', 'SVD', 'Round_3', 'Beta_1', 'beta_1', [.1, .2, .3, .4, .5, .6, .7, .8, .9])

# Networks Using Same Settings as Project 1 for Baseline Scores. #
# pcaReduced = cs.Classifier("PCA Reduction", neural_network.MLPClassifier(activation='identity', solver='sgd', beta_1=0.4), pca_features, labels)
# icaReduced = cs.Classifier("ICA Reduction", neural_network.MLPClassifier(activation='identity', solver='adam', beta_1=0.9), ica_features, labels)
# rpReduced = cs.Classifier("RP Reduction", neural_network.MLPClassifier(activation='identity', solver='sgd', beta_1=0.4), rp_features, labels)
# svdReduced = cs.Classifier("SVD Reduction", neural_network.MLPClassifier(activation='identity', solver='adam', beta_1=0.9), svd_features, labels)
# baseLine = cs.Classifier("Baseline", neural_network.MLPClassifier(activation='identity', solver='adam', beta_1=0.4), features, labels)

# classifiers = [pcaReduced,
#                 icaReduced,
#                 rpReduced,
#                 svdReduced,
#                 baseLine]

# plotting.plot_roc_curves_dimensionality_reduction(classifiers, "Dimensionality Reduced Neural Networks - Tuned", 'Charts/Titanic/' + dataset_name + "_dimensionality_reduced_neural_networks_roc_curve.png")

###########################################################
# Part 5. Run Neural Networks with Clustering as feature. #
###########################################################

# Optimal clusters for KMeans - 4. #
kmeansOptimalModel, kmeans_Features = analysis.get_kmeans_clustering_predictions(features, 4)
kmeans_Features = kmeans_Features.reshape(-1, 1)

# Optimal clusters for Expectations Maximization - 4. #
emOptimalModel, em_Features = analysis.get_expectations_maximization_predictions(features, 4)
em_Features = em_Features.reshape(-1, 1)

# Tune KMeans. #
#analysis.run_analysis(neural_network.MLPClassifier(), kmeans_Features, labels, 'Titanic', 'KMeans', 'Round_1', 'Activation', 'activation', ['identity', 'logistic', 'tanh', 'relu'])
#analysis.run_analysis(neural_network.MLPClassifier(activation='relu'), kmeans_Features, labels, 'Titanic', 'KMeans', 'Round_2', 'Solver', 'solver', ['lbfgs', 'sgd', 'adam'])
#analysis.run_analysis(neural_network.MLPClassifier(activation='relu', solver='adam'), kmeans_Features, labels, 'Titanic', 'KMeans', 'Round_3', 'Beta_1', 'beta_1', [.1, .2, .3, .4, .5, .6, .7, .8, .9])

# Tune Expectation Maxmization. #
#analysis.run_analysis(neural_network.MLPClassifier(), em_Features, labels, 'Titanic', 'EM', 'Round_1', 'Activation', 'activation', ['identity', 'logistic', 'tanh', 'relu'])
#analysis.run_analysis(neural_network.MLPClassifier(activation='relu'), em_Features, labels, 'Titanic', 'EM', 'Round_2', 'Solver', 'solver', ['lbfgs', 'sgd', 'adam'])
#analysis.run_analysis(neural_network.MLPClassifier(activation='relu', solver='adam'), em_Features, labels, 'Titanic', 'EM', 'Round_3', 'Beta_1', 'beta_1', [.1, .2, .3, .4, .5, .6, .7, .8, .9])

# Networks Using Same Settings as Project 1 for Baseline Scores. #
kmeansNN = cs.Classifier("KMeans", neural_network.MLPClassifier(activation='relu', solver='adam', beta_1=0.5), kmeans_Features, labels)
emNN = cs.Classifier("Expectation Maximization", neural_network.MLPClassifier(activation='relu', solver='adam', beta_1=0.4), em_Features, labels)
baseLineNN = cs.Classifier("Baseline", neural_network.MLPClassifier(activation='identity', solver='adam', beta_1=0.4), features, labels)

classifiers = [kmeansNN,
                emNN,
                baseLineNN]

plotting.plot_roc_curves_dimensionality_reduction(classifiers, "Clustering Neural Networks - Tuned", 'Charts/Titanic/' + dataset_name + "_clustered_neural_networks_roc_curve.png")
