#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import numpy as np
import matplotlib.pyplot as plt

# 1. Retrieve and load the Olivetti faces dataset
faces_data = fetch_olivetti_faces(shuffle=True, random_state=42)

# X contains the image data, y contains the labels
X = faces_data.data
y = faces_data.target

# 2. Split the Dataset using Stratified Sampling
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# 3. Apply PCA
n_components = 40
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)
X_test_pca = pca.transform(X_test)

# Define the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Define the K-fold cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
cv_scores_rf = cross_val_score(rf_classifier, X_train_pca, y_train, cv=kf, scoring='accuracy')

print("Random Forest Cross-validation scores:", cv_scores_rf)
print("Mean cross-validation score:", np.mean(cv_scores_rf))

# Train the Random Forest Classifier on the full training set
rf_classifier.fit(X_train_pca, y_train)

# Evaluate the Random Forest classifier on the validation set
y_val_pred_rf = rf_classifier.predict(X_val_pca)

# Print the classification report
print(classification_report(y_val, y_val_pred_rf))

# Plot the training data with PCA 
def plot_pca_2d(X, y, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=30)
    plt.title(title)
    plt.colorbar(label='Class')
    plt.show()

# Reduce PCA to 2D for visualization
pca_2d = PCA(n_components=2)
X_train_pca_2d = pca_2d.fit_transform(X_train_pca)

plot_pca_2d(X_train_pca_2d, y_train, "PCA Projection (First Classifier)")

# 4. Agglomerative Hierarchical Clustering (AHC) and Dendrogram Plotting

# Function to plot a dendrogram
def plot_dendrogram(Z, title):
    plt.figure(figsize=(10, 5))
    dendrogram(Z)
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.show()

# Function to compute and plot silhouette scores for different numbers of clusters
def compute_silhouette(X, dist_matrix, linkage_method, max_clusters=150):
    best_score = -1
    best_n_clusters = 2
    best_labels = None
    
    for n_clusters in range(2, max_clusters + 1):
        # Apply Agglomerative Clustering
        cluster = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage=linkage_method)
        cluster_labels = cluster.fit_predict(squareform(dist_matrix))
        
        # Compute silhouette score
        score = silhouette_score(X, cluster_labels)
        
        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters
            best_labels = cluster_labels
            
    print(f"Best silhouette score: {best_score} with {best_n_clusters} clusters.")
    return best_n_clusters, best_labels


# 4(a): Euclidean Distance
print("\nClustering using Euclidean Distance and Silhouette Score")

# Compute the Euclidean distance matrix
euclidean_dist_matrix = pdist(X_train_pca, metric='euclidean')

# Perform linkage and plot dendrogram for Euclidean distance
Z_euclidean = linkage(euclidean_dist_matrix, method='average')
plot_dendrogram(Z_euclidean, "Dendrogram (Euclidean Distance)")

# Find the best number of clusters using silhouette score for Euclidean distance
best_clusters_euclidean, euclidean_labels = compute_silhouette(X_train_pca, euclidean_dist_matrix, linkage_method='average')


# 4(b): Manhattan Distance (Minkowski with p=1)
print("\nClustering using Manhattan Distance and Silhouette Score")

# Compute the Manhattan (Minkowski, p=1) distance matrix
manhattan_dist_matrix = pdist(X_train_pca, metric='minkowski', p=1)

# Perform linkage and plot dendrogram for Manhattan distance
Z_manhattan = linkage(manhattan_dist_matrix, method='average')
plot_dendrogram(Z_manhattan, "Dendrogram (Manhattan Distance)")

# Find the best number of clusters using silhouette score for Manhattan distance
best_clusters_manhattan, manhattan_labels = compute_silhouette(X_train_pca, manhattan_dist_matrix, linkage_method='average')


# 4(c): Cosine Similarity
print("\nClustering using Cosine Similarity and Silhouette Score")

# Compute the Cosine distance matrix
cosine_dist_matrix = pdist(X_train_pca, metric='cosine')

# Perform linkage and plot dendrogram for Cosine similarity using 'average' linkage
Z_cosine = linkage(cosine_dist_matrix, method='average')  
plot_dendrogram(Z_cosine, "Dendrogram (Cosine Similarity)")

# Find the best number of clusters using silhouette score for Cosine similarity
best_clusters_cosine, cosine_labels = compute_silhouette(X_train_pca, cosine_dist_matrix, linkage_method='average')

# Function to plot PCA projection with cluster labels
def plot_pca_clusters(X_pca, labels, title):
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', s=30)
    plt.title(title)
    plt.colorbar(scatter, label='Cluster')
    plt.show()

# Plot PCA projection with Euclidean distance-based clustering
plot_pca_clusters(X_train_pca_2d, euclidean_labels, "PCA Projection with Euclidean Clustering")

# 6. Train a Classifier using the Cluster Labels from 4(a)

# Use cluster labels as features for classification
X_train_with_clusters = np.hstack([X_train_pca, euclidean_labels.reshape(-1, 1)])  # Add the cluster labels to features

# Compute cluster labels for the validation set using the same clustering method
# Ensure the number of clusters does not exceed the number of validation samples
n_clusters_to_use = min(best_clusters_euclidean, X_val.shape[0])

val_dist_matrix = pdist(X_val_pca, metric='euclidean')
val_cluster_labels = AgglomerativeClustering(n_clusters=n_clusters_to_use, metric='precomputed', linkage='average').fit_predict(squareform(val_dist_matrix))

# Append the cluster labels to the PCA-transformed validation set
X_val_with_clusters = np.hstack([X_val_pca, val_cluster_labels.reshape(-1, 1)])  # Add the cluster labels to validation features

# Define the Random Forest Classifier again
rf_classifier_with_clusters = RandomForestClassifier(n_estimators=100, random_state=42)

# Perform cross-validation with the clustered data
cv_scores_rf_clusters = cross_val_score(rf_classifier_with_clusters, X_train_with_clusters, y_train, cv=kf, scoring='accuracy')

print("Random Forest (with Clusters) Cross-validation scores:", cv_scores_rf_clusters)
print("Mean cross-validation score (with clusters):", np.mean(cv_scores_rf_clusters))

# Train the classifier on the training set with cluster features
rf_classifier_with_clusters.fit(X_train_with_clusters, y_train)

# Evaluate the classifier on the validation set with cluster features
y_val_pred_rf_clusters = rf_classifier_with_clusters.predict(X_val_with_clusters)

# Print the classification report
print(classification_report(y_val, y_val_pred_rf_clusters))

# Plot the training data with cluster features
X_train_with_clusters_pca_2d = pca_2d.transform(X_train_with_clusters[:, :-1])  
plot_pca_2d(X_train_with_clusters_pca_2d, y_train, "PCA Projection (After Adding Clusters)")
