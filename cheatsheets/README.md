# 📝 ML Models Interview Cheatsheets

Welcome to the cheatsheets folder! This directory is designed as a high-density, quick-access reference guide for your machine learning engineering and data science interview preparation.

Each file contains the essential math, scikit-learn syntax, from-scratch implementation snippets, complexity parameters, and common interview questions/answers for all the key algorithms in our curriculum.

---

## 🗺️ Cheatsheet Map

| Index | Category | Algorithms Covered | Cheatsheet Link |
|---|---|---|---|
| **01** | **Regression** | Linear, Polynomial, Ridge, Lasso, Elastic Net | [01_regression.md](01_regression.md) |
| **02** | **Classification** | Logistic Regression, Naive Bayes, SVM, Decision Trees, k-NN | [02_classification.md](02_classification.md) |
| **03** | **Clustering** | K-Means, DBSCAN, Hierarchical Clustering, GMM, Mean Shift | [03_clustering.md](03_clustering.md) |
| **04** | **Dimensionality Reduction** | PCA, t-SNE, UMAP, SVD, LDA | [04_dimensionality_reduction.md](04_dimensionality_reduction.md) |
| **05** | **Association Rule Learning** | Apriori, Eclat, FP-Growth | [05_association_rules.md](05_association_rules.md) |
| **06** | **Ensemble Learning** | Random Forest, AdaBoost, GBM, XGBoost, LightGBM, CatBoost, Stacking/Voting | [06_ensemble_learning.md](06_ensemble_learning.md) |

---

## ⚡ High-Yield Cheat Table (Complexities & Attributes)

| Algorithm | Training Time Complexity | Prediction Time Complexity | Space Complexity | Parametric / Non-Parametric | Key Hyperparameters |
|---|---|---|---|---|---|
| **Linear Regression** | $O(d^2 n + d^3)$ (Normal Eq) or $O(k n d)$ | $O(d)$ | $O(d)$ | Parametric | Learning rate $\alpha$, epochs |
| **Logistic Regression**| $O(k n d)$ | $O(d)$ | $O(d)$ | Parametric | Penalty ($L_1$/$L_2$), $C$ (Inverse regularization) |
| **Naive Bayes** | $O(n d)$ | $O(c d)$ | $O(c d)$ | Parametric | Laplace smoothing $\alpha$ |
| **SVM** | $O(n^2 d)$ to $O(n^3 d)$ | $O(n_{sv} d)$ | $O(n_{sv} d)$ | Non-Parametric | $C$, kernel, gamma, degree |
| **Decision Trees** | $O(d n \log n)$ | $O(\text{depth})$ | $O(\text{nodes})$ | Non-Parametric | `max_depth`, `min_samples_split`, `criterion` |
| **k-NN** | $O(1)$ (Lazy) | $O(n d)$ | $O(n d)$ | Non-Parametric | $k$, distance metric, weights |
| **K-Means** | $O(t \cdot k \cdot n \cdot d)$ | $O(k \cdot d)$ | $O(k \cdot d)$ | Non-Parametric | `n_clusters`, `init` |
| **DBSCAN** | $O(n \log n)$ (with Index) or $O(n^2)$ | $O(1)$ (assigning query) | $O(n)$ | Non-Parametric | `eps`, `min_samples` |
| **PCA** | $O(d^2 n + d^3)$ or $O(k^2 n + k^3)$ | $O(k d)$ | $O(k d)$ | Parametric | `n_components`, `svd_solver` |
| **Apriori** | Exponential $O(2^d)$ | N/A | $O(2^d)$ | Non-Parametric | `min_support`, `min_confidence` |
| **Random Forest** | $O(M \cdot k \cdot n \log n)$ | $O(M \cdot \text{depth})$ | $O(M \cdot \text{nodes})$ | Non-Parametric | `n_estimators`, `max_features`, `max_depth` |
| **XGBoost / GBM** | $O(M \cdot d \cdot n \log n)$ | $O(M \cdot \text{depth})$ | $O(M \cdot \text{nodes})$ | Non-Parametric | `learning_rate`, `max_depth`, `subsample`, `reg_lambda` |

*Note: $n$ = number of samples, $d$ = number of features, $k$ = target components / clusters / neighbors, $c$ = number of classes, $n_{sv}$ = number of support vectors, $t$ = iterations.*

---

## 💡 Top 3 Cheat Rules for Interviews

1. **Always scale features** for distance-based algorithms (k-NN, K-Means, SVM, PCA). It is **not** strictly required for tree-based algorithms (Decision Trees, Random Forest).
2. **If dataset is high-dimensional ($d > n$)**, linear SVM or Logistic Regression with L1 regularization (Lasso) are excellent choices.
3. **If training data cannot fit in memory**, use mini-batch gradient descent (e.g., SGDClassifier, MiniBatchKMeans) or online learning.
