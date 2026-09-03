#!/usr/bin/env python3
"""Check distribution examples, estimation and the exercise contracts."""

import importlib.util
from pathlib import Path
import unittest


PROJECT = Path(__file__).resolve().parents[1] / 'math/probability'


def load(name, class_name):
    """Load a submitted class without modifying the import path."""
    spec = importlib.util.spec_from_file_location(
        name, PROJECT / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


class ProbabilityTests(unittest.TestCase):
    """Compare independently known examples and domain boundaries."""

    @classmethod
    def setUpClass(cls):
        """Load the four standalone distribution classes."""
        cls.Poisson = load('poisson', 'Poisson')
        cls.Exponential = load('exponential', 'Exponential')
        cls.Normal = load('normal', 'Normal')
        cls.Binomial = load('binomial', 'Binomial')

    def test_constructor_errors_and_empty_data(self):
        """An empty supplied sample must not select the default parameters."""
        for distribution in (self.Poisson, self.Exponential,
                             self.Normal, self.Binomial):
            with self.subTest(distribution=distribution.__name__):
                with self.assertRaisesRegex(TypeError, 'data must be a list'):
                    distribution((1, 2))
                for data in ([], [1]):
                    with self.assertRaisesRegex(
                            ValueError, '^data must contain multiple values$'):
                        distribution(data)
        for distribution in (self.Poisson, self.Exponential):
            for rate in (0, -1):
                with self.assertRaisesRegex(ValueError, 'positive value'):
                    distribution(lambtha=rate)
        with self.assertRaisesRegex(ValueError, 'stddev must be a positive'):
            self.Normal(stddev=0)
        with self.assertRaisesRegex(ValueError, 'n must be a positive'):
            self.Binomial(n=0)
        for probability in (0, 1, -0.1, 1.1):
            with self.assertRaisesRegex(ValueError, 'p must be greater'):
                self.Binomial(p=probability)

    def test_parameter_estimation_and_types(self):
        """Check population variance and the binomial method of moments."""
        self.assertEqual(self.Poisson([2, 4]).lambtha, 3.0)
        self.assertEqual(self.Exponential([.25, .75]).lambtha, 2.0)
        normal = self.Normal([2, 4])
        self.assertEqual((normal.mean, normal.stddev), (3.0, 1.0))
        binomial = self.Binomial([1, 2, 2, 3])
        self.assertEqual(binomial.n, 3)
        self.assertAlmostEqual(binomial.p, 2 / 3)
        self.assertIsInstance(self.Poisson(lambtha=5).lambtha, float)
        self.assertIsInstance(self.Normal(mean=2).mean, float)
        self.assertIsInstance(self.Binomial(n=3).n, int)

    def test_poisson_published_values_and_support(self):
        """The cumulative mass includes all integer counts through k."""
        poisson = self.Poisson(lambtha=5)
        self.assertAlmostEqual(poisson.pmf(9), 0.036265577412911795)
        self.assertAlmostEqual(poisson.cdf(9), 0.9681719426208609)
        self.assertEqual(poisson.pmf(3.9), poisson.pmf(3))
        self.assertEqual(poisson.cdf(3.9), poisson.cdf(3))
        self.assertEqual(poisson.pmf(-1), 0)
        self.assertEqual(poisson.cdf(-1), 0)
        self.assertEqual(poisson.cdf(0), poisson.pmf(0))

    def test_exponential_published_values_and_support(self):
        """Exponential density and cumulative probability start at zero."""
        exponential = self.Exponential(lambtha=2)
        self.assertAlmostEqual(exponential.pdf(1), 0.2706705664650693)
        self.assertAlmostEqual(exponential.cdf(1), 0.8646647167674654)
        self.assertEqual(exponential.pdf(0), 2.0)
        self.assertEqual(exponential.cdf(0), 0.0)
        self.assertEqual(exponential.pdf(-1), 0)
        self.assertEqual(exponential.cdf(-1), 0)

    def test_normal_published_approximation_and_round_trip(self):
        """Use the stipulated erf polynomial rather than math.erf."""
        normal = self.Normal(mean=70, stddev=10)
        self.assertEqual(normal.z_score(90), 2.0)
        self.assertEqual(normal.x_value(2), 90.0)
        self.assertAlmostEqual(normal.pdf(90), 0.005399096651147344)
        self.assertAlmostEqual(normal.cdf(90), 0.9922398930659416)
        self.assertEqual(normal.cdf(70), .5)
        self.assertAlmostEqual(normal.cdf(50) + normal.cdf(90), 1.0)
        self.assertEqual(normal.x_value(normal.z_score(82)), 82)

    def test_binomial_published_values_and_exercise_support(self):
        """Follow the explicit zero result outside the binomial support."""
        binomial = self.Binomial(n=50, p=.6)
        self.assertAlmostEqual(binomial.pmf(30), 0.114558552829524)
        self.assertAlmostEqual(binomial.cdf(30), 0.5535236207894576)
        self.assertAlmostEqual(binomial.cdf(50), 1.0)
        for k in (-1, 51):
            self.assertEqual(binomial.pmf(k), 0)
            self.assertEqual(binomial.cdf(k), 0)
        self.assertEqual(binomial.pmf(30.7), binomial.pmf(30))
        self.assertEqual(binomial.cdf(30.7), binomial.cdf(30))


if __name__ == '__main__':
    unittest.main()
