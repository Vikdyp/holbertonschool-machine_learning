# Clustering

K-means initialization, Lloyd updates, within-cluster variance and elbow
comparison; Gaussian mixture initialization, density evaluation, expectation
maximization and BIC selection; scikit-learn wrappers and Ward hierarchical
clustering with a colored dendrogram.

The NumPy implementations follow Holberton project 2330's import and loop
constraints. Gaussian covariance updates use maximum-likelihood normalization;
PDF values are bounded below by 1e-300. Invalid parameters and singular
Gaussian distributions return the failure values specified by each exercise.
The scikit-learn wrappers retain the library's default model settings.
