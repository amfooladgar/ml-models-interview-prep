# 🛒 Association Rule Learning Cheatsheet

Covers: **Apriori, Eclat, FP-Growth**

---

## 1. Core Metrics

Association rule learning finds interesting relations between items in large transaction databases (e.g., Market Basket Analysis).

Given a rule: $A \rightarrow B$ (e.g., if a customer buys Bread, they also buy Butter):
- **Support**: The fraction of total transactions that contain both items.
  $$\text{Support}(A \rightarrow B) = \frac{\text{Count}(A \cap B)}{N}$$
- **Confidence**: How often the items in $B$ appear in transactions that contain $A$.
  $$\text{Confidence}(A \rightarrow B) = \frac{\text{Support}(A \rightarrow B)}{\text{Support}(A)} = \frac{\text{Count}(A \cap B)}{\text{Count}(A)}$$
- **Lift**: Measures how much more often $A$ and $B$ occur together than expected if they were statistically independent.
  $$\text{Lift}(A \rightarrow B) = \frac{\text{Support}(A \rightarrow B)}{\text{Support}(A) \cdot \text{Support}(B)}$$
  - $\text{Lift} > 1$: Positive correlation (Items are bought together more often than by chance).
  - $\text{Lift} = 1$: Independent.
  - $\text{Lift} < 1$: Negative correlation (Substitute items; buying one makes buying the other less likely).

---

## 2. Apriori

### Core Intuition
An iterative algorithm that finds frequent itemsets using the **Apriori Property** (any subset of a frequent itemset must also be frequent; if an itemset is infrequent, all of its supersets will also be infrequent).

### Algorithm Steps
1. Find all frequent 1-itemsets (support $\geq$ `min_support`).
2. Join frequent $k$-itemsets to generate candidate $(k+1)$-itemsets.
3. Prune candidate sets that contain any infrequent subsets.
4. Scan the database to calculate support for remaining candidates, keep those with support $\geq$ `min_support`.
5. Repeat until no new frequent itemsets are found.

### Python Syntax (`mlxtend`)
```python
from mlxtend.frequent_patterns import apriori, association_rules
# X is a one-hot encoded transaction DataFrame
frequent_itemsets = apriori(df_onehot, min_support=0.05, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
```

---

## 3. Eclat

### Core Intuition
Equivalence Class Clustering and bottom-up Lattice Traversal. Eclat uses a **vertical database format** (item-to-transaction mappings, i.e., TID-lists) instead of horizontal (transaction-to-item). It finds frequent itemsets by taking intersections of TID-lists, which is much faster than scanning the entire transaction database.

### Python / Custom Syntax
```python
# Custom vertical format intersection logic:
# support of itemset {A, B} is the size of intersection of TID-list of A and TID-list of B.
tid_list_A = {1, 3, 4, 7}
tid_list_B = {1, 2, 4, 8}
tid_list_AB = tid_list_A.intersection(tid_list_B) # {1, 4} -> support is 2
```

---

## 4. FP-Growth

### Core Intuition
Frequent Pattern Growth. Solves the bottleneck of Apriori (frequent database scans and massive candidate generation) by compressing the transaction database into a compact tree structure called an **FP-Tree**. It then extracts frequent itemsets directly from the tree recursively.

### Key Concepts
- **FP-Tree**: Root is null. Transactions are inserted as prefix paths. Nodes store item IDs and item counts.
- **Header Table**: Linked list pointing to all occurrences of each item in the tree to facilitate quick traversals.
- **Conditional FP-Tree**: Formed by selecting prefix paths for a suffix item, updating counts, and building a tree.

### Python Syntax (`mlxtend`)
```python
from mlxtend.frequent_patterns import fpgrowth, association_rules
frequent_itemsets = fpgrowth(df_onehot, min_support=0.05, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
```

---

## 🚀 High-Yield Interview Questions

### Q: Why is FP-Growth faster than Apriori?
**Answer**: Apriori requires scanning the database multiple times (once for each itemset size $k$) and generating a massive number of candidate itemsets, which takes significant memory and disk I/O. FP-Growth scans the database only twice: first to count 1-itemset support, and second to construct the compact FP-Tree. It then mines frequent patterns directly from the tree without any candidate generation.

### Q: If support(A) = 0.4, support(B) = 0.5, and support(A $\cap$ B) = 0.1, what is the lift of the rule A $\rightarrow$ B?
**Answer**:
$$\text{Lift}(A \rightarrow B) = \frac{\text{Support}(A \rightarrow B)}{\text{Support}(A) \cdot \text{Support}(B)} = \frac{0.1}{0.4 \cdot 0.5} = \frac{0.1}{0.2} = 0.5$$
Since the Lift is less than 1, buying A and buying B are negatively correlated.
