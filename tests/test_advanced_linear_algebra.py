#!/usr/bin/env python3
"""Test exact matrix examples, inverse identities and definiteness."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1] / 'math/advanced_linear_algebra'


def load(index, name):
    """Load an exercise by its numbered filename."""
    path = ROOT / '{}-{}.py'.format(index, name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)


class MatrixTests(unittest.TestCase):
    """Verify algebraic results without relying on implementation details."""

    @classmethod
    def setUpClass(cls):
        """Load the six submitted functions."""
        cls.functions = [load(i, name) for i, name in enumerate(
            ['determinant', 'minor', 'cofactor', 'adjugate', 'inverse',
             'definiteness'])]

    def test_determinant_and_empty_convention(self):
        """Check exact scalar, singular and three-dimensional results."""
        determinant = self.functions[0]
        self.assertEqual(determinant([[]]), 1)
        self.assertEqual(determinant([[5]]), 5)
        self.assertEqual(determinant([[1, 2], [3, 4]]), -2)
        self.assertEqual(determinant([[1, 1], [1, 1]]), 0)
        self.assertEqual(determinant([[5, 7, 9], [3, 1, 8], [6, 2, 4]]), 192)

    def test_minors_cofactors_and_adjugates(self):
        """Use independently calculated quiz examples and one-cell matrices."""
        minor, cofactor, adjugate = self.functions[1:4]
        self.assertEqual(minor([[-7, 0, 6], [5, -2, -10], [4, 3, 2]]),
                         [[26, 50, 23], [-18, -38, -21], [12, 40, 14]])
        self.assertEqual(cofactor([[6, -9, 9], [7, 5, 0], [4, 3, -8]]),
                         [[-40, 56, 1], [-45, -84, -54], [-45, 63, 93]])
        self.assertEqual(adjugate([[-4, 1, 9], [-9, -8, -5], [-3, 8, 10]]),
                         [[-40, 62, 67], [105, -13, -101], [-96, 29, 41]])
        for function in (minor, cofactor, adjugate):
            self.assertEqual(function([[5]]), [[1]])

    def test_inverse_identity_singular_and_input_preservation(self):
        """A returned inverse must undo the original matrix on both sides."""
        inverse = self.functions[4]
        matrix = [[1, 0, 1], [2, 1, 2], [1, 0, -1]]
        saved = [row[:] for row in matrix]
        result = inverse(matrix)
        np.testing.assert_allclose(np.matmul(matrix, result), np.eye(3))
        np.testing.assert_allclose(np.matmul(result, matrix), np.eye(3))
        self.assertEqual(matrix, saved)
        self.assertIsNone(inverse([[1, 1], [1, 1]]))
        self.assertIsNone(inverse([[0]]))
        self.assertEqual(inverse([[4]]), [[0.25]])

    def test_invalid_python_matrix_shapes(self):
        """Distinguish malformed list containers from nonsquare matrices."""
        for function in self.functions[:5]:
            for matrix in (None, [], [1, 2], [[1], (2,)]):
                with self.assertRaisesRegex(TypeError, 'list of lists'):
                    function(matrix)
            for matrix in ([[1, 2]], [[1], [2, 3]]):
                with self.assertRaisesRegex(ValueError, 'square matrix'):
                    function(matrix)
        for function in self.functions[1:5]:
            with self.assertRaisesRegex(ValueError, 'non-empty square matrix'):
                function([[]])

    def test_all_definiteness_categories(self):
        """Cover each sign pattern and relative numerical scaling."""
        definiteness = self.functions[5]
        examples = [([[5, 1], [1, 1]], 'Positive definite'),
                    ([[2, 4], [4, 8]], 'Positive semi-definite'),
                    ([[-1, 1], [1, -1]], 'Negative semi-definite'),
                    ([[-2, 4], [4, -9]], 'Negative definite'),
                    ([[1, 2], [2, 1]], 'Indefinite')]
        for matrix, expected in examples:
            self.assertEqual(definiteness(np.array(matrix)), expected)
        self.assertEqual(definiteness(np.eye(2) * 1e-20), 'Positive definite')
        self.assertEqual(definiteness(np.diag([1e20, -1.0])), 'Indefinite')
        self.assertEqual(definiteness(np.zeros((2, 2))),
                         'Positive semi-definite')

    def test_invalid_numpy_matrices(self):
        """Reject empty, nonsquare, nonsymmetric and nonfinite inputs."""
        definiteness = self.functions[5]
        with self.assertRaisesRegex(TypeError, 'must be a numpy.ndarray'):
            definiteness([[1]])
        for matrix in (np.array([]), np.ones((2, 3)),
                       np.array([[1, 2], [0, 1]]), np.array([[np.nan]])):
            self.assertIsNone(definiteness(matrix))


if __name__ == '__main__':
    unittest.main()
