# 🏷️ Classification

Classification algorithms predict a **discrete category or class label** (e.g., Spam vs Not Spam, Disease vs Healthy, Digit 0-9).

## Algorithms in This Section

| # | Algorithm | Key Concepts | Difficulty |
|---|-----------|-------------|------------|
| 1 | [Logistic Regression](01-logistic-regression/) | Sigmoid function, Log-Loss/Binary Cross-Entropy, Decision Boundary, ROC-AUC | ⭐ Beginner |
| 2 | [Naive Bayes](02-naive-bayes/) | Bayes Theorem, Conditional Independence assumption, Gaussian vs Multinomial vs Bernoulli NB, Laplace Smoothing | ⭐ Beginner |
| 3 | [Support Vector Machines (SVM)](03-svm/) | Maximum Margin, Support Vectors, Hard vs Soft Margin (C parameter), Kernel Trick (RBF/Polynomial), Hinge Loss | ⭐⭐ Intermediate |
| 4 | [Decision Trees](04-decision-trees/) | Entropy, Information Gain, Gini Impurity, CART algorithm, Tree Pruning, Feature Importance | ⭐ Beginner |
| 5 | [k-Nearest Neighbors (k-NN)](05-knn/) | Lazy learning, Distance Metrics (Euclidean, Manhattan, Minkowski), Curse of Dimensionality, Choice of K | ⭐ Beginner |

## Evaluation Metrics for Classification

| Metric | Formula / Meaning | Best Used When |
|--------|-------------------|----------------|
| **Accuracy** | `(TP + TN) / Total` | Balanced classes |
| **Precision** | `TP / (TP + FP)` | High cost of False Positives (e.g., Spam detection) |
| **Recall (Sensitivity)** | `TP / (TP + FN)` | High cost of False Negatives (e.g., Cancer detection) |
| **F1-Score** | `2 * (Precision * Recall) / (Precision + Recall)` | Imbalanced dataset, trade-off between Precision and Recall |
| **ROC-AUC** | Area under TPR vs FPR curve | Model evaluation across all decision thresholds |
| **Confusion Matrix** | Breakdown of TP, FP, TN, FN | Detailed error analysis |

## Learning Path

```
Logistic Regression → Naive Bayes → SVM → Decision Trees → k-NN
  (Linear boundary)   (Probabilistic)  (Max margin)  (Tree splits)  (Instance-based)
```
