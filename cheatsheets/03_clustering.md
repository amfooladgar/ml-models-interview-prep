# 👥 Clustering Cheatsheet

Covers: **K-Means, DBSCAN, Hierarchical Clustering, Gaussian Mixture Models (GMM), Mean Shift**

---

## 1. K-Means

### Core Intuition
An iterative clustering algorithm that partitions data into $k$ clusters. Each point is assigned to its nearest centroid, and centroids are re-calculated as the mean of their points until convergence.

### Mathematics
- **Objective Function (Inertia/WCSS)**:
  $$J = \sum_{j=1}^k \sum_{i \in S_j} \|x^{(i)} - \mu_j\|^2$$
  where $\mu_j$ is the centroid of cluster $S_j$.
- **K-Means++ Initialization**: Selects initial centroids that are far apart from each other probabilistically (reduces the chance of finding sub-optimal local minima).

### Scikit-Learn Syntax
```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, random_state=42)
labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_
inertia = kmeans.inertia_
```

---

## 2. DBSCAN

### Core Intuition
Density-Based Spatial Clustering of Applications with Noise. Group points that are close together in high-density regions, and marks sparse-region points as outliers. Does not require pre-specifying $k$ and can find arbitrary cluster shapes.

### Mathematics / Key Concepts
- **$\epsilon$ (epsilon)**: The maximum radius of a neighborhood.
- **`min_samples`**: Minimum number of points in an $\epsilon$-neighborhood to classify a point as a Core point.
- **Point Categories**:
  - *Core Point*: $\geq$ `min_samples` points within its $\epsilon$-neighborhood.
  - *Border Point*: $<$ `min_samples` points, but falls within the $\epsilon$-neighborhood of a Core point.
  - *Noise Point*: Neither a Core nor a Border point.

### Scikit-Learn Syntax
```python
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)  # Outliers are labeled as -1
```

---

## 3. Hierarchical Clustering (Agglomerative)

### Core Intuition
Builds a tree of clusters (dendrogram) bottom-up by starting with each point as its own cluster and iteratively merging the closest pair of clusters.

### Linkage Criteria
Determines how distance between two clusters is measured:
- **Single Linkage**: Minimum distance between any point in cluster A and cluster B.
- **Complete Linkage**: Maximum distance between any point in cluster A and cluster B.
- **Average Linkage**: Average distance between all pairs of points.
- **Ward Linkage**: Minimizes the total variance (sum of squared errors) within clusters.

### Scikit-Learn Syntax
```python
from sklearn.cluster import AgglomerativeClustering
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = agg.fit_predict(X)
```

---

## 4. Gaussian Mixture Models (GMM)

### Core Intuition
A probabilistic clustering model assuming that all data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters. Solved using the Expectation-Maximization (EM) algorithm. GMM is a soft clustering method.

### Mathematics
- **Probability Density**: $p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)$ where $\sum \pi_k = 1$.
- **Expectation-Maximization (EM)**:
  - *E-Step*: Calculate posterior probabilities (responsibilities) of points belonging to each component.
  - *M-Step*: Update mean $\mu_k$, covariance $\Sigma_k$, and weight $\pi_k$ of components using the soft assignments.

### Scikit-Learn Syntax
```python
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=3, covariance_type='full') # covariance types: 'spherical', 'tied', 'diag', 'full'
gmm.fit(X)
labels = gmm.predict(X)
probs = gmm.predict_proba(X)  # Soft assignment matrix
```

---

## 5. Mean Shift

### Core Intuition
A non-parametric clustering algorithm that identifies dense regions of points by shifting centroids towards the mode (highest density peak) of the distribution, using kernel density estimation.

### Mathematics / Update Rule
- **Mean Shift Vector**:
  $$m(x) = \frac{\sum_{i=1}^n x^{(i)} K(\frac{x^{(i)} - x}{h})}{\sum_{i=1}^n K(\frac{x^{(i)} - x}{h})} - x$$
  where $K$ is the kernel (usually Gaussian) and $h$ is the bandwidth.
- Points update their positions $x \leftarrow x + m(x)$ until convergence. Points that converge to the same peak belong to the same cluster.

### Scikit-Learn Syntax
```python
from sklearn.cluster import MeanShift, estimate_bandwidth
bandwidth = estimate_bandwidth(X, quantile=0.2)
ms = MeanShift(bandwidth=bandwidth)
labels = ms.fit_predict(X)
```

---

## 🚀 High-Yield Interview Questions

### Q: Compare K-Means and DBSCAN. When would you use one over the other?
**Answer**:
- **K-Means** requires specifying $k$, assumes spherical/convex cluster shapes, is sensitive to outliers, and cannot handle varying densities well. It is fast, scaling as $O(n)$.
- **DBSCAN** does not require specifying $k$, can identify arbitrary/complex shapes, explicitly flags noise/outliers, and handles density well. It is slower ($O(n^2)$ or $O(n \log n)$) and struggles with high dimensions or varying densities.
- Use **DBSCAN** if you suspect arbitrary shapes or noise/outliers (e.g., spatial telemetry). Use **K-Means** for general partitioning when clusters are roughly spherical and speed is critical.

### Q: What is the difference between hard and soft clustering? Give examples.
**Answer**:
- **Hard clustering** assigns each data point to exactly one cluster (probability of 0 or 1). Examples: K-Means, DBSCAN, Hierarchical Clustering.
- **Soft clustering** assigns each point a probability or membership score for belonging to each cluster. Example: Gaussian Mixture Models (GMM), where we get an $n \times k$ matrix of component probabilities.
