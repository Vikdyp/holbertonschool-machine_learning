#!/usr/bin/env python3
"""Fit a Gaussian mixture using scikit-learn."""
import sklearn.mixture


def gmm(X, k):
    """Return mixture parameters, predicted labels and the fitted BIC."""
    model = sklearn.mixture.GaussianMixture(n_components=k).fit(X)
    return (model.weights_, model.means_, model.covariances_,
            model.predict(X), model.bic(X))
