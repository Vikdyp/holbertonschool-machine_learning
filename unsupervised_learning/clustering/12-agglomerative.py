#!/usr/bin/env python3
"""Display and cut a Ward-linkage hierarchical clustering tree."""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """Show cluster-colored dendrogram and return distance-cut labels."""
    linkage = scipy.cluster.hierarchy.linkage(X, method='ward')
    scipy.cluster.hierarchy.dendrogram(linkage, color_threshold=dist)
    plt.show()
    return scipy.cluster.hierarchy.fcluster(linkage, dist,
                                            criterion='distance')
