#!/usr/bin/env python3
"""Fit K-means using scikit-learn."""
import sklearn.cluster


def kmeans(X, k):
    """Return fitted scikit-learn centroids and cluster assignments."""
    model = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    return model.cluster_centers_, model.labels_
