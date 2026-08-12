# 🌲 Ensemble Learning Cheatsheet

Covers: **Bagging, Random Forest, AdaBoost, Gradient Boosting Machine (GBM), XGBoost, LightGBM, CatBoost, Stacking & Voting Ensembles**

---

## 1. Overview & Paradigm Comparison

Ensemble methods combine multiple base models (weak learners) to construct a stronger predictive model with better generalization and reduced variance/bias.

| Paradigm | Primary Goal | Sequential or Parallel? | Key Algorithm Examples |
|---|---|---|---|
| **Bagging** (Bootstrap Aggregation) | Reduce Variance (Overfitting) | Parallel | Random Forest, ExtraTrees, Bagging Classifier |
| **Boosting** | Reduce Bias (Underfitting) | Sequential | AdaBoost, GBM, XGBoost, LightGBM, CatBoost |
| **Stacking / Voting** | Combine Diverse Hypotheses | Independent / Layered | Voting Classifier, Stacking Classifier (Meta-Learner) |

---

## 2. Bagging & Random Forest

### Core Intuition
- **Bagging**: Trains multiple base estimators independently on bootstrap samples (random sampling with replacement) and aggregates predictions (majority vote for classification, average for regression).
- **Random Forest**: Extends Bagging by injecting **feature randomness**. At each split node, only a random subset of features ($k = \sqrt{d}$ for classification, $k = d/3$ for regression) is considered.

### Mathematics
- **Variance Reduction**: If $M$ independent models have variance $\sigma^2$, their average has variance $\frac{\sigma^2}{M}$. If models are correlated with pairwise correlation $\rho$, variance is:
  $$\text{Var}(\bar{X}) = \rho \sigma^2 + \frac{1-\rho}{M} \sigma^2$$
  *Random Forest reduces $\rho$ by randomly sampling features at each split.*
- **Out-of-Bag (OOB) Error**: About $1 - \frac{1}{e} \approx 36.8\%$ of samples are left out of each bootstrap sample and serve as a free validation set.

### Scikit-Learn Syntax
```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    max_features='sqrt',
    oob_score=True,
    random_state=42
)
rf.fit(X_train, y_train)
oob_acc = rf.oob_score_
```

---

## 3. AdaBoost (Adaptive Boosting)

### Core Intuition
Sequentially trains weak learners (typically decision stumps — trees of depth 1). In each iteration, sample weights are increased for misclassified instances, forcing the next weak learner to focus on hard samples.

### Mathematics
- **Sample Weight Update**:
  $$w_i^{(t+1)} = w_i^{(t)} \cdot \exp\left(-\alpha_t y_i h_t(x_i)\right)$$
- **Learner Weight ($\alpha_t$)**:
  $$\alpha_t = \frac{1}{2} \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)$$
  where $\epsilon_t$ is the weighted error rate of decision stump $h_t$.

### Scikit-Learn Syntax
```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

adaboost = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)
adaboost.fit(X_train, y_train)
```

---

## 4. Gradient Boosting Machine (GBM)

### Core Intuition
Fits sequential weak learners to the **pseudo-residuals** (negative gradients of the loss function) of the previous ensemble's predictions.

### Mathematics
- **Residual Calculation**:
  $$r_{i,t} = -\left[ \frac{\partial L(y_i, f(x_i))}{\partial f(x_i)} \right]_{f(x_i) = f_{t-1}(x_i)}$$
- **Ensemble Update**:
  $$f_t(x) = f_{t-1}(x) + \eta \cdot \gamma_t h_t(x)$$
  where $\eta$ is the learning rate (shrinkage).

### Scikit-Learn Syntax
```python
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
gbm = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    random_state=42
)
gbm.fit(X_train, y_train)
```

---

## 5. XGBoost (Extreme Gradient Boosting)

### Core Intuition
An optimized, highly scalable implementation of gradient boosting that uses 2nd-order Taylor expansions of the loss function, regularization terms, weighted quantile sketches, and parallel hardware optimization.

### Mathematics
- **Objective Function at step $t$**:
  $$\mathcal{L}^{(t)} \approx \sum_{i=1}^n \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t)$$
  where $g_i = \frac{\partial L}{\partial \hat{y}^{(t-1)}}$ (first derivative/gradient), $h_i = \frac{\partial^2 L}{\partial (\hat{y}^{(t-1)})^2}$ (second derivative/Hessian), and $\Omega(f) = \gamma T + \frac{1}{2} \lambda \sum w_j^2$.

### Library Syntax
```python
import xgboost as xgb
model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,   # L1 regularization
    reg_lambda=1.0,  # L2 regularization
    random_state=42
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
```

---

## 6. LightGBM

### Core Intuition
Developed by Microsoft for high efficiency. Key innovations:
1. **Leaf-wise Tree Growth** (grows node with max delta loss vs level-wise).
2. **GOSS (Gradient-based One-Side Sampling)**: Keeps all instances with large gradients, randomly samples instances with small gradients.
3. **EFB (Exclusive Feature Bundling)**: Bundles mutually exclusive sparse features into dense features.

### Library Syntax
```python
import lightgbm as lgb
model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.05,
    num_leaves=31,     # Main structural parameter (controls complexity)
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)
```

---

## 7. CatBoost

### Core Intuition
Developed by Yandex. Optimized for categorical features:
1. **Target Encoding / Ordered Target Statistics**: Computes target encoding on random permutations of data without target leakage.
2. **Oblivious / Symmetric Trees**: Uses the same split feature across all nodes at a given depth, making predictions lightning-fast and hardware-friendly.

### Library Syntax
```python
from catboost import CatBoostClassifier
cat_features_idx = [0, 2] # Index of categorical columns
model = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    cat_features=cat_features_idx,
    verbose=0
)
model.fit(X_train, y_train)
```

---

## 8. Stacking & Voting Ensembles

### Core Intuition
- **Voting**: Combines predictions from multiple distinct algorithms via hard voting (majority rule) or soft voting (average predicted probabilities).
- **Stacking**: Trains a meta-learner (e.g. Logistic Regression / Ridge) using out-of-fold predictions from several diverse base models.

### Scikit-Learn Syntax
```python
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Voting
voting = VotingClassifier(
    estimators=[('lr', LogisticRegression()), ('dt', DecisionTreeClassifier()), ('svc', SVC(probability=True))],
    voting='soft'
)

# Stacking
stacking = StackingClassifier(
    estimators=[('rf', RandomForestClassifier()), ('gbm', GradientBoostingClassifier())],
    final_estimator=LogisticRegression(),
    cv=5
)
```

---

## 🚀 High-Yield Interview Questions

### Q: Compare Random Forest and XGBoost. When would you choose one over the other?
**Answer**:
- **Random Forest** trains independent deep trees in parallel (Bagging), reducing variance. It rarely overfits, requires minimal hyperparameter tuning, and handles noisy data well.
- **XGBoost** trains shallow trees sequentially (Boosting), reducing bias and variance. It usually achieves higher predictive accuracy on clean tabular data but is sensitive to hyperparameters and noisy labels.
- Choose **Random Forest** for quick baseline models with minimal tuning. Choose **XGBoost/LightGBM/CatBoost** for competitive performance and production tabular applications.

### Q: How does LightGBM achieve much faster training speeds than standard Gradient Boosting?
**Answer**:
LightGBM uses:
1. **Histogram-based algorithms**: Buckets continuous features into discrete bins ($O(n \cdot d) \to O(\text{bins} \cdot d)$).
2. **GOSS**: Focuses computation on instances with large gradients while down-sampling small-gradient instances.
3. **EFB**: Reduces feature dimensionality by bundling exclusive sparse features.
4. **Leaf-wise tree growth**: Splits nodes with maximum loss reduction instead of expanding level-by-level.
