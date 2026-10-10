# Dimensionality Reduction

PCA projections and the stages of a NumPy t-SNE implementation: distances,
conditional Gaussian affinities, Student-t affinities, gradients, KL divergence
and momentum optimization.

The numbered functions follow the interfaces of Holberton project 2329. Task 0
selects components by cumulative singular values, as shown by its example.
This convention differs from the usual explained-variance ratio based on their
squares. The gradient deliberately omits the scalar factor four required to be
omitted by the exercise. Entropy is measured in bits; KL divergence uses natural
logarithms. Early exaggeration lasts 100 updates and momentum changes after 20.
The entropy search rejects unattainable perplexities or numerical stagnation
rather than silently returning affinities outside the requested tolerance.

Algorithm reference: [Visualizing Data using t-SNE](https://jmlr.org/papers/v9/vandermaaten08a.html).
