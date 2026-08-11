# 📈 Regression Cheatsheet

Covers: **Linear Regression, Polynomial Regression, Ridge/Lasso/ElasticNet Regression**

---

## 1. Linear Regression

### Core Intuition
Models a linear relationship between input features $X$ and a continuous target $y$ by minimizing the sum of squared differences (residuals).

### Mathematics
- **Hypothesis Function**: $\hat{y} = X\theta$ (where $X$ includes a column of 1s for the bias/intercept $\theta_0$).
- **Cost Function (Mean Squared Error)**:
  $$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2 = \frac{1}{2m} (X\theta - y)^T(X\theta - y)$$
- **Gradient Descent Update**:
  $$\theta := \theta - \alpha \frac{1}{m} X^T(X\theta - y)$$
- **Normal Equation (Closed-Form)**:
  $$\theta = (X^T X)^{-1} X^T y$$

### Scikit-Learn Syntax
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
coef = model.coef_
intercept = model.intercept_
```

### From-Scratch Snippet (NumPy)
```python
# Normal Equation
X_b = np.c_[np.ones((len(X), 1)), X]  # Add bias column
theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
```

---

## 2. Polynomial Regression

### Core Intuition
Extends linear models to fit non-linear relationships by creating polynomial features from the original inputs.

### Mathematics
- **Feature Transformation**: Transform $x$ into $[1, x, x^2, x^3, \dots, x^d]$.
- **Hypothesis**: $y = \theta_0 + \theta_1 x + \theta_2 x^2 + \dots + \theta_d x^d$.
- Since it remains linear in terms of the parameters $\theta$, it is still solved using the linear regression algorithms.

### Scikit-Learn Syntax
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

poly_model = make_pipeline(
    PolynomialFeatures(degree=3, include_bias=False),
    LinearRegression()
)
poly_model.fit(X_train, y_train)
```

---

## 3. Ridge & Lasso & Elastic Net

### Core Intuition
Regularization techniques to prevent overfitting by penalizing large weights.
- **Ridge (L2)**: Shrinks weights close to 0; good when many features contribute to target.
- **Lasso (L1)**: Shrinks weights to exactly 0; performs feature selection.
- **Elastic Net**: Combination of L1 and L2 regularization.

### Mathematics
- **Ridge Cost Function**:
  $$J(\theta) = \text{MSE}(\theta) + \alpha \frac{1}{2} \sum_{i=1}^n \theta_i^2$$
- **Lasso Cost Function**:
  $$J(\theta) = \text{MSE}(\theta) + \alpha \sum_{i=1}^n |\theta_i|$$
- **Elastic Net Cost Function**:
  $$J(\theta) = \text{MSE}(\theta) + r \cdot \alpha \sum_{i=1}^n |\theta_i| + \frac{1-r}{2} \cdot \alpha \sum_{i=1}^n \theta_i^2$$

### Scikit-Learn Syntax
```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
# alpha represents the regularization strength (commonly lambda in literature)
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=1.0)
elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)  # l1_ratio = r
```

---

## 🚀 High-Yield Interview Questions

### Q: Why does L1 regularization (Lasso) lead to sparse feature weights, while L2 (Ridge) does not?
**Answer**: Geometrically, the constraint region for L1 is a diamond shape (with sharp corners on the axes), while L2 is a hypersphere. When optimizing, the contours of the loss function are highly likely to hit the sharp corners of the L1 diamond on the axes first, forcing some coordinates to be exactly zero. The L2 circle has no corners, so weights are shrunk close to zero but rarely set to exactly zero.

### Q: What are the main assumptions of Linear Regression?
**Answer**:
1. **L**inearity: The relationship between independent and dependent variables is linear.
2. **I**ndependence: Residuals must be independent (no autocorrelation).
3. **N**ormality: Residuals must follow a normal distribution centered at 0.
4. **E**qual Variance (Homoscedasticity): The variance of the residuals remains constant across all predictions.
