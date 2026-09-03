# Probability

Standalone Python implementations without imported modules:

| File | Distribution | Operations |
| --- | --- | --- |
| `poisson.py` | Poisson | Rate estimation, PMF, CDF |
| `exponential.py` | Exponential | Rate estimation, PDF, CDF |
| `normal.py` | Normal | Population moments, z-score conversions, PDF, CDF |
| `binomial.py` | Binomial | Method of moments, PMF, CDF |

The implementations use the required approximations for pi, e and erf.
The degree-nine erf polynomial is only an approximation near the mean;
it is not suitable for accurate normal tail probabilities. The binomial CDF
returns zero outside `[0, n]`, as explicitly requested by the exercise.

From the repository root, run the local contract tests with:

```sh
python -m unittest discover -s tests -p test_probability.py
```
