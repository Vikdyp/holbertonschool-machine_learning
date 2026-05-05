#!/usr/bin/env python3
"""Random forest classifier built from decision trees."""

import numpy as np

Decision_Tree = __import__('8-build_decision_tree').Decision_Tree


class Random_Forest:
    """Random forest model using random decision trees."""

    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """Initialize a Random_Forest instance."""
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed

    def predict(self, explanatory):
        """Predict classes by majority vote over all trees."""
        predictions = np.array([
            predict(explanatory)
            for predict in self.numpy_preds
        ])
        return np.array([
            np.bincount(predictions[:, i].astype(int)).argmax()
            for i in range(explanatory.shape[0])
        ])

    def fit(self, explanatory, target, n_trees=100, verbose=0):
        """Fit the random forest on explanatory data and targets."""
        self.target = target
        self.explanatory = explanatory
        self.numpy_preds = []
        self.n_trees = n_trees
        depths = []
        nodes = []
        leaves = []
        accuracies = []
        for i in range(n_trees):
            T = Decision_Tree(max_depth=self.max_depth,
                              min_pop=self.min_pop,
                              seed=self.seed + i)
            T.fit(explanatory, target)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
            accuracies.append(T.accuracy(T.explanatory, T.target))
        if verbose == 1:
            print(f"""  Training finished.
    - Mean depth                     : {np.array(depths).mean()}
    - Mean number of nodes           : {np.array(nodes).mean()}
    - Mean number of leaves          : {np.array(leaves).mean()}
    - Mean accuracy on training data : {np.array(accuracies).mean()}
    - Accuracy of the forest on td   : {self.accuracy(self.explanatory,
                                                      self.target)}""")

    def accuracy(self, test_explanatory, test_target):
        """Return the accuracy of the forest on test data."""
        return np.sum(
            np.equal(self.predict(test_explanatory), test_target)
        ) / test_target.size
