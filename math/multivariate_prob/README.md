# Multivariate Probability

Estimate multivariate sample means and covariance matrices, convert covariance
to correlation, and evaluate the density of a multivariate normal distribution.

| File | Description |
| --- | --- |
| `0-mean_cov.py` | Row mean and unbiased covariance for observations shaped `(n, d)` |
| `1-correlation.py` | Correlation matrix from a square covariance matrix |
| `multinormal.py` | Gaussian parameters from `(d, n)` observations and density at `(d, 1)` |

The submissions use only NumPy and calculate covariance directly from centered
observations, with the sample normalization `n - 1`.
