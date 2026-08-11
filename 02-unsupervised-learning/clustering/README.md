# 👥 Clustering

Clustering algorithms group data points such that points in the same group (cluster) are more similar to each other than to those in other groups.

## Algorithms in This Section

| # | Algorithm | Key Concepts | Difficulty |
|---|-----------|-------------|------------|
| 1 | [K-Means](01-kmeans/) | Centroids, WCSS, Elbow Method, K-Means++ | ⭐ Beginner |
| 2 | [DBSCAN](02-dbscan/) | Density, Epsilon, Min Samples, Core/Border/Noise | ⭐⭐ Intermediate |
| 3 | [Hierarchical Clustering](03-hierarchical/) | Dendrogram, Linkages (Ward, Single, Complete) | ⭐⭐ Intermediate |
| 4 | [Gaussian Mixture Models](04-gmm/) | Soft clustering, Gaussian Mixture, EM Algorithm | ⭐⭐⭐ Advanced |
| 5 | [Mean Shift](05-meanshift/) | Non-parametric, Bandwidth, Mode-seeking | ⭐⭐⭐ Advanced |

---

## Evaluation Metrics

- **Silhouette Coefficient**: Measures how similar a point is to its own cluster compared to other clusters. Ranges from -1 to 1.
- **Inertia (WCSS)**: Sum of squared distances of samples to their closest cluster center. Used in K-Means.
- **Davies-Bouldin Index**: Average similarity measure of each cluster with its most similar cluster. Lower score is better.
