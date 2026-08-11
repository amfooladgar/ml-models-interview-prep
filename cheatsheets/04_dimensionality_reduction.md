# 📐 Dimensionality Reduction Cheatsheet

Covers: **Principal Component Analysis (PCA), t-SNE, UMAP, Singular Value Decomposition (SVD), Linear Discriminant Analysis (LDA)**

---

## 1. Principal Component Analysis (PCA)

### Core Intuition
An unsupervised linear technique that projects high-dimensional data onto a lower-dimensional subspace of orthogonal directions (principal components) that maximize the variance of the projected data.

### Mathematics
1. Standardize the data: $X \leftarrow \frac{X - \mu}{\sigma}$.
2. Compute the Covariance Matrix: $\Sigma = \frac{1}{m} X^T X$.
3. Compute eigenvectors and eigenvalues: $\Sigma v = \lambda v$.
4. Sort eigenvectors by eigenvalues (descending order) and choose the top $k$.
5. Project data: $X_{\text{reduced}} = X \cdot W$ (where $W$ contains the top $k$ eigenvectors).

### Scikit-Learn Syntax
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)
explained_variance_ratio = pca.explained_variance_ratio_
```

---

## 2. t-SNE

### Core Intuition
t-Distributed Stochastic Neighbor Embedding. A non-linear, probabilistic technique primarily used for visualization. Maps high-dimensional data to a low-dimensional space by preserving pairwise local similarities.

### Mathematics / Key Concepts
- Computes conditional probabilities that represent similarities between points in the high-dimensional space (using Gaussian distribution) and low-dimensional space (using Student-t distribution).
- **Student-t distribution** in low-dimensional space solves the **crowding problem** (moderate distances in high dimensions collapse to tiny distances in low dimensions).
- Minimizes the Kullback-Leibler (KL) divergence between high and low dimensional probability distributions using gradient descent:
  $$\text{KL}(P || Q) = \sum_{i} \sum_{j} p_{j|i} \log \frac{p_{j|i}}{q_{j|i}}$$

### Scikit-Learn Syntax
```python
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=30.0, learning_rate='auto', random_state=42)
X_embedded = tsne.fit_transform(X)
```

---

## 3. UMAP

### Core Intuition
Uniform Manifold Approximation and Projection. A non-linear dimensionality reduction technique based on Riemannian geometry and algebraic topology. It is much faster than t-SNE while preserving both local and global structure better.

### Scikit-Learn / Library Syntax
*UMAP is usually imported from the third-party `umap-learn` library.*
```python
import umap
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
X_embedded = reducer.fit_transform(X)
```

---

## 4. Singular Value Decomposition (SVD)

### Core Intuition
A fundamental matrix factorization technique that decomposes a matrix into three matrices, revealing geometric structure. Truncated SVD is used to perform dimensionality reduction (specifically on sparse datasets like text term-document matrices, also known as LSA).

### Mathematics
- Decomposes any $m \times n$ real matrix $A$ into:
  $$A = U \Sigma V^T$$
  - $U$ ($m \times m$): Left singular vectors (orthonormal).
  - $\Sigma$ ($m \times n$): Singular values (diagonal, non-negative).
  - $V^T$ ($n \times n$): Right singular vectors (orthonormal).
- Truncated SVD keeps only the $k$ largest singular values to approximate $A$.

### Scikit-Learn Syntax
```python
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=5, algorithm='randomized', random_state=42)
X_reduced = svd.fit_transform(X_sparse)
```

---

## 5. Linear Discriminant Analysis (LDA)

### Core Intuition
Unlike PCA, LDA is a **supervised** dimensionality reduction technique. It projects features into a lower-dimensional space to maximize class separability (maximizing between-class variance while minimizing within-class variance).

### Mathematics
- Finds projection vector $w$ that maximizes Fisher's criterion:
  $$J(w) = \frac{w^T S_B w}{w^T S_W w}$$
  - $S_B$ (Between-class scatter matrix): Measures distance between class means.
  - $S_W$ (Within-class scatter matrix): Measures variance within each class.
- The maximum number of components is limited to $C - 1$ (where $C$ is the number of classes).

### Scikit-Learn Syntax
```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=1) # Max components: min(d, classes-1)
X_lda = lda.fit_transform(X, y)
```

---

## 🚀 High-Yield Interview Questions

### Q: What is the main difference between PCA and LDA?
**Answer**:
- **PCA** is unsupervised. It finds the directions of maximum variance in the data without considering labels.
- **LDA** is supervised. It finds the directions that maximize class separability using class labels.
- If classification is the end-goal, LDA is often better; however, if the dataset is small or class distributions are complex, PCA might generalize better to avoid overfitting.

### Q: Explain the difference between PCA and t-SNE. When would you use which?
**Answer**:
- **PCA** is linear, global, and highly scalable. It preserves global variances and pairwise distances. It is fast and deterministic.
- **t-SNE** is non-linear, local, and slow. It preserves local neighborhood structures but does not preserve global distances or densities. It is stochastic and cannot be easily applied to new out-of-sample data points (`fit_transform` must be run on the entire dataset).
- Use **t-SNE** exclusively for visualizing high-dimensional datasets in 2D or 3D. Use **PCA** for pre-processing, noise reduction, and feature size compression before training models.
