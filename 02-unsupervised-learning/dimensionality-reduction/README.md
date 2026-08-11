# 📐 Dimensionality Reduction

Dimensionality reduction algorithms compress feature spaces by mapping high-dimensional spaces to lower-dimensional manifolds, helping with visualization, noise reduction, and pre-processing.

## Algorithms in This Section

| # | Algorithm | Key Concepts | Difficulty |
|---|-----------|-------------|------------|
| 1 | [PCA](01-pca/) | Eigenvectors, Covariance Matrix, Variance Maximization | ⭐ Beginner |
| 2 | [t-SNE](02-tsne/) | KL Divergence, Perplexity, Student-t distribution | ⭐⭐⭐ Advanced |
| 3 | [UMAP](03-umap/) | Manifold learning, Simplicial complexes, Speed | ⭐⭐⭐ Advanced |
| 4 | [Singular Value Decomposition](04-svd/) | Matrix Factorization, Singular Values, Latent Semantics | ⭐⭐ Intermediate |
| 5 | [Linear Discriminant Analysis](05-lda/) | Supervised projection, Class Separability, Scatter Matrices | ⭐⭐ Intermediate |

---

## Linear vs Non-Linear Reduction

- **Linear Reduction** (PCA, SVD, LDA) assumes the data lies on a linear subspace. They preserve global structures (distance, variance) and are highly scalable.
- **Non-Linear Reduction** (t-SNE, UMAP) maps complex non-linear manifolds. They preserve local neighbor structures but may distort global distance metrics. Best suited for low-dimensional visualization.
