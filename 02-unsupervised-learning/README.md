# 🌀 Unsupervised Learning

Unsupervised learning algorithms find hidden patterns, structures, or relationships in unlabeled datasets. 

This phase covers three main subcategories: **Clustering**, **Dimensionality Reduction**, and **Association Rule Learning**.

---

## 📁 Subcategory Directory

### 👥 [Clustering](clustering/)
Group similar data points together based on distance, density, or probabilistic distributions.
- **K-Means** — Centroid-based partitioning
- **DBSCAN** — Density-based spatial clustering
- **Hierarchical Clustering** — Agglomerative tree-based merging
- **Gaussian Mixture Models (GMM)** — Probabilistic soft clustering
- **Mean Shift** — Non-parametric mode-seeking

### 📐 [Dimensionality Reduction](dimensionality-reduction/)
Compress high-dimensional feature spaces while preserving essential data structures or class separations.
- **Principal Component Analysis (PCA)** — Linear variance maximization
- **t-SNE** — Local neighborhood similarity preservation (visualization-focused)
- **UMAP** — Topological manifold approximation
- **Singular Value Decomposition (SVD)** — Matrix factorization
- **Linear Discriminant Analysis (LDA)** — Supervised class separability maximization

### 🛒 [Association Rule Learning](association-rule-learning/)
Discover interesting purchasing relations or conditional co-occurrences between items in transaction logs.
- **Apriori** — Breadth-first candidate generation & pruning
- **Eclat** — Vertical database layout set intersection
- **FP-Growth** — FP-tree compression and pattern mining

---

## ⚡ Quick Guide & Best Practices

1. **Feature Scaling is Mandatory** for distance-based clustering (K-Means, Hierarchical) and dimensionality reduction (PCA). Scaling is not required for association rule learning.
2. **Evaluation Metrics**:
   - Clustering: Silhouette Score, Davies-Bouldin Index, Inertia (Elbow Method).
   - Dimensionality Reduction: Explained Variance Ratio, Reconstruction Loss, KL Divergence.
   - Association Rules: Support, Confidence, Lift.
