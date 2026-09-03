# Advanced Linear Algebra

Six standalone exercises cover determinants, minor matrices, cofactors,
adjugates, inverses and definiteness.

Tasks 0–4 use pure Python without imported modules. The determinant of `[[]]`
is one; singular matrices have no inverse and return `None`. The functions
preserve the input matrix. Task 5 uses the signs of NumPy eigenvalues to classify
real symmetric matrices.

Run the contract tests from the repository root:

```sh
python -m unittest discover -s tests -p test_advanced_linear_algebra.py
```
