# 🌲 Ensemble Learning

Ensemble Learning techniques combine predictions from multiple individual models (weak learners) to build a robust model with lower variance, lower bias, or superior accuracy.

## Algorithms in This Section

| # | Algorithm | Key Paradigm | Key Concepts | Difficulty |
|---|-----------|-------------|--------------|------------|
| 1 | [Random Forest & Bagging](01-random-forest/) | Bagging | Parallel trees, Bootstrap Aggregation, Feature Sub-sampling, OOB Score | ⭐ Beginner |
| 2 | [AdaBoost](02-adaboost/) | Boosting | Sequential decision stumps, Sample weight update, Exponential loss | ⭐⭐ Intermediate |
| 3 | [Gradient Boosting Machine](03-gbm/) | Boosting | Residual fitting, Gradient descent in function space, Learning rate | ⭐⭐ Intermediate |
| 4 | [XGBoost](04-xgboost/) | Boosting | 2nd-order Taylor expansion, L1/L2 tree regularization, Quantile sketch | ⭐⭐⭐ Advanced |
| 5 | [LightGBM](05-lightgbm/) | Boosting | Leaf-wise tree growth, GOSS, Exclusive Feature Bundling (EFB) | ⭐⭐⭐ Advanced |
| 6 | [CatBoost](06-catboost/) | Boosting | Ordered Target Statistics, Target leakage prevention, Oblivious trees | ⭐⭐⭐ Advanced |
| 7 | [Stacking & Voting](07-stacking-voting/) | Hybrid / Meta | Hard vs Soft Voting, Out-of-fold predictions, Meta-Learner | ⭐⭐ Intermediate |

---

## ⚡ Key Paradigm Summary

- **Bagging** (Random Forest): Builds independent trees in parallel on bootstrap samples. Reduces **variance** (overfitting).
- **Boosting** (AdaBoost, GBM, XGBoost, LightGBM, CatBoost): Builds sequential trees, each correcting errors/residuals of prior trees. Reduces **bias** and **variance**.
- **Stacking**: Combines predictions from diverse base algorithms using a meta-learner model trained on out-of-fold cross-validation predictions.
