# 📘 Supervised Learning

Supervised learning is the most widely used form of machine learning. The model learns from **labeled data** — input-output pairs where the correct answer is known — to make predictions on unseen data.

## Key Concepts

- **Training Data**: A dataset of (input, label) pairs
- **Loss Function**: Measures how wrong the model's predictions are
- **Optimization**: Minimizes the loss function (e.g., gradient descent)
- **Generalization**: The ability to perform well on unseen data
- **Overfitting vs. Underfitting**: The fundamental tradeoff

## Categories

### 📈 [Regression](regression/)
Predict a **continuous** output variable.

| Algorithm | Notebook | Key Idea |
|-----------|----------|----------|
| Linear Regression | [Practice](regression/01-linear-regression/) | Fit a straight line to data |
| Polynomial Regression | [Practice](regression/02-polynomial-regression/) | Fit curves using polynomial features |
| Ridge/Lasso Regression | [Practice](regression/03-ridge-lasso-regression/) | Regularized regression to prevent overfitting |

### 🏷️ [Classification](classification/)
Predict a **discrete** class label.

| Algorithm | Notebook | Key Idea |
|-----------|----------|----------|
| Logistic Regression | *Phase 2* | Linear decision boundary with sigmoid |
| Naive Bayes | *Phase 2* | Probabilistic classifier using Bayes' theorem |
| SVM | *Phase 2* | Maximum margin classifier |
| Decision Trees | *Phase 2* | Tree-based recursive partitioning |
| k-NN | *Phase 2* | Classify based on nearest neighbors |

## Interview Tips

1. **Always discuss the bias-variance tradeoff** when comparing models
2. **Know the assumptions** of each algorithm (e.g., linear regression assumptions)
3. **Be ready to derive** gradient descent for linear/logistic regression
4. **Understand regularization** — it comes up in almost every ML interview
5. **Know when to use what** — there's no universally best algorithm
