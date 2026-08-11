# 🏷️ Classification Cheatsheet

Covers: **Logistic Regression, Naive Bayes, Support Vector Machines (SVM), Decision Trees, k-Nearest Neighbors (k-NN)**

---

## 1. Logistic Regression

### Core Intuition
Models the probability that a sample belongs to a binary class using the Sigmoid (Logistic) function. Solved using maximum likelihood estimation.

### Mathematics
- **Sigmoid (Logistic) Function**: $\sigma(z) = \frac{1}{1 + e^{-z}}$ where $z = \theta^T x$.
- **Hypothesis Function**: $h_\theta(x) = \sigma(\theta^T x) = P(y=1|x;\theta)$.
- **Binary Cross-Entropy Loss (Log-Loss)**:
  $$J(\theta) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1-y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right]$$

### Scikit-Learn Syntax
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(penalty='l2', C=1.0)  # C is inverse regularization strength
model.fit(X_train, y_train)
probs = model.predict_proba(X_test)[:, 1]  # Get probability of class 1
```

---

## 2. Naive Bayes

### Core Intuition
A probabilistic classifier based on Bayes' Theorem with the "naive" assumption of conditional independence between features given the class label.

### Mathematics
- **Bayes' Theorem**: $P(y|x_1, \dots, x_d) \propto P(y) \prod_{j=1}^d P(x_j|y)$.
- **Laplace Smoothing**: $\theta_{y,i} = \frac{N_{yi} + \alpha}{N_y + \alpha \cdot d}$ (prevents zero probability).
- **Types**:
  - *Gaussian NB*: Continuous features ($P(x_i|y)$ is modeled via normal distribution).
  - *Multinomial NB*: Count data (e.g., text classifications).
  - *Bernoulli NB*: Binary features.

### Scikit-Learn Syntax
```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
```

---

## 3. Support Vector Machines (SVM)

### Core Intuition
Finds the optimal hyperplane that maximizes the margin (distance between the hyperplane and closest points of any class, which are support vectors).

### Mathematics
- **Margin Maximization**: Minimize $\frac{1}{2} \|\mathbf{w}\|^2$ subject to $y^{(i)}(\mathbf{w}^T x^{(i)} + b) \geq 1$.
- **Soft Margin (Hinge Loss)**: $\min \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^m \xi^{(i)}$.
- **Kernel Trick (Dual Form)**: Replaces dot products with kernel functions $K(x^{(i)}, x^{(j)})$ to compute similarity in a high-dimensional space without explicitly mapping points. E.g., RBF kernel: $K(x, z) = \exp(-\gamma \|x - z\|^2)$.

### Scikit-Learn Syntax
```python
from sklearn.svm import SVC
# kernel options: 'linear', 'poly', 'rbf', 'sigmoid'
clf = SVC(C=1.0, kernel='rbf', gamma='scale')
clf.fit(X_train, y_train)
```

---

## 4. Decision Trees

### Core Intuition
Recursively splits features into nodes to maximize the homogeneity (purity) of resulting child leaves.

### Mathematics
- **Entropy**: $H(S) = -\sum p_i \log_2 p_i$.
- **Gini Impurity**: $G(S) = 1 - \sum p_i^2$.
- **Information Gain**: $\text{IG}(S, A) = H(S) - \sum \frac{|S_v|}{|S|} H(S_v)$.

### Scikit-Learn Syntax
```python
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_split=2)
tree.fit(X_train, y_train)
importances = tree.feature_importances_
```

---

## 5. k-Nearest Neighbors (k-NN)

### Core Intuition
Instance-based, lazy learner that classifies a query point by a majority vote of its $k$ nearest neighbors in feature space.

### Mathematics
- **Distance Metrics**:
  - *Euclidean ($L_2$)*: $d(x, z) = \sqrt{\sum (x_i - z_i)^2}$.
  - *Manhattan ($L_1$)*: $d(x, z) = \sum |x_i - z_i|$.
- **Curse of Dimensionality**: High dimensions dilate space, causing distance metrics to lose discriminative power as all points appear equidistant.

### Scikit-Learn Syntax
```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2) # p=2 is Euclidean
knn.fit(X_train, y_train)
```

---

## 🚀 High-Yield Interview Questions

### Q: Why do tree-based classifiers not require feature scaling?
**Answer**: Decision tree splits are monotonic transformations. A split on feature $X_j$ at value $s$ splits the data into $X_j \leq s$ and $X_j > s$. Multiplying all values of $X_j$ by a positive scaling factor $c$ simply changes the split value to $c \cdot s$, leaving the split subsets and tree structure identical.

### Q: What is the effect of changing parameter C in SVM?
**Answer**: $C$ controls the tradeoff between model simplicity and training error. A **large $C$** penalizes classification errors heavily, resulting in a narrow margin that tries to fit training data perfectly (prone to **overfitting**). A **small $C$** allows more training errors, leading to a wider margin (prone to **underfitting**).
