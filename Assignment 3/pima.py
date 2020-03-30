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
pima = pd.read_csv("PimaIndianDiabetes/diabetes.csv")
dataset_name = 'Pima'

features = pima.iloc[:,:-1]
labels = np.array(pima.iloc[:,-1:])

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
pca = PCA()
pca_features = pca.fit_transform(features)

plotting.plot_pca_eigen_values(dataset_name, dataset_name + '/pca_baseline.png', pca, features)

########
# LCA. #
########
ica = FastICA(max_iter=5000, tol=.15)
ica_features = ica.fit_transform(features)

plotting.plot_ica_kurtosis(dataset_name, dataset_name + '/ica_baseline.png', ica, ica_features, features)

###########################
# Randomized Projections. #
###########################
rp = GaussianRandomProjection(n_components=8)
rp_features = rp.fit_transform(features)

plotting.plot_random_projections(dataset_name, dataset_name + '/rp_baseline.png', rp, rp_features, features)

# ##################
# # Truncated SVD. #
# ##################
svd = TruncatedSVD()
svd_features = svd.fit_transform(features)

plotting.plot_svd(dataset_name, dataset_name + '/svd_baseline.png', svd, svd_features, features)

# ###########################################################
# # Part 3. Clustering with Dimensionally Reduced Features. #
# ###########################################################

# analysis.run_clustering_algorithms(pca_features, labels, dataset_name, 'pca_reduced')
# analysis.run_clustering_algorithms(ica_features, labels, dataset_name, 'ica_reduced')
# analysis.run_clustering_algorithms(rp_features, labels, dataset_name, 'randomized_projections_reduced')
# analysis.run_clustering_algorithms(svd_features, labels, dataset_name, 'svd_reduced')