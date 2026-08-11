# 🛒 Association Rule Learning

Association rule learning discovers interesting purchase relationships or conditional co-occurrences between items in transaction databases.

## Algorithms in This Section

| # | Algorithm | Key Concepts | Difficulty |
|---|-----------|-------------|------------|
| 1 | [Apriori](01-apriori/) | Candidate generation, Support/Confidence, Pruning | ⭐⭐ Intermediate |
| 2 | [Eclat](02-eclat/) | Vertical database layout, TID-lists, Set Intersections | ⭐⭐ Intermediate |
| 3 | [FP-Growth](03-fpgrowth/) | FP-Tree, Prefix paths, Conditional FP-Tree | ⭐⭐⭐ Advanced |

---

## Core Association Metrics

Given a rule: $A \rightarrow B$
- **Support**: Fraction of total transactions containing both $A$ and $B$.
- **Confidence**: Probability of item $B$ being purchased given item $A$ is purchased.
- **Lift**: Ratio of observed support of $A$ and $B$ together to that expected if they were independent. Lift $> 1$ indicates positive association.
