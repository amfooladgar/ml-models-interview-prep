# 📈 Regression

Regression algorithms predict a **continuous numerical value**. They are among the most fundamental ML concepts and are frequently asked about in interviews.

## Algorithms in This Section

| # | Algorithm | Key Concepts | Difficulty |
|---|-----------|-------------|------------|
| 1 | [Linear Regression](01-linear-regression/) | OLS, gradient descent, normal equation, R² | ⭐ Beginner |
| 2 | [Polynomial Regression](02-polynomial-regression/) | Feature engineering, bias-variance, overfitting | ⭐ Beginner |
| 3 | [Ridge/Lasso Regression](03-ridge-lasso-regression/) | L1/L2 regularization, feature selection, elastic net | ⭐⭐ Intermediate |

## Prerequisites

- Basic linear algebra (vectors, matrices, dot products)
- Calculus (derivatives, partial derivatives, chain rule)
- Python + NumPy fundamentals

## Learning Path

```
Linear Regression → Polynomial Regression → Ridge/Lasso Regression
     (basics)          (complexity)           (regularization)
```

Each notebook builds on the previous one. Start with Linear Regression and work through sequentially.

## Common Interview Topics

- Deriving the normal equation
- Implementing gradient descent from scratch
- Explaining R², adjusted R², MSE, MAE, RMSE
- Bias-variance tradeoff visualization
- When and why to regularize
- L1 vs L2 regularization geometric intuition
- Multicollinearity and VIF
- Assumptions of linear regression (LINE: Linearity, Independence, Normality, Equal variance)

## Evaluation Metrics

| Metric | Formula | Use Case |
|--------|---------|----------|
| MSE | `mean((y - ŷ)²)` | Penalizes large errors more |
| RMSE | `sqrt(MSE)` | Same units as target |
| MAE | `mean(|y - ŷ|)` | Robust to outliers |
| R² | `1 - SS_res/SS_tot` | Proportion of variance explained |
| Adjusted R² | `1 - (1-R²)(n-1)/(n-p-1)` | Accounts for number of features |
