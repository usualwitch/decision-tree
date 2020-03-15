# Decision Tree

An implementation of simplified C4.5 and CART algorithms.

A 10-fold CV is applied to evaluate the performance of algorithms.

- [Decision Tree](#decision-tree)
  - [Usage](#usage)
  - [Splitting](#splitting)
  - [Pruning](#pruning)
    - [C4.5](#c45)
    - [CART](#cart)
  - [Reference](#reference)
  - [License](#license)

## Usage

Clone the repo and run the code.

```python
from tree import DecisionTree


data = pd.read_csv('data/knowledge.csv')
config = {'algorithm': 'C4.5', 'penalty_coeffs': , 'penalty_funcs': }
dt = DecisionTree(data)
print(dt.root)
```

## Splitting

Both C4.5 and CART generate trees by recursive partitioning, continuing to subdivide the set of training cases until each subset in the partition contains cases of a single class, or until no split offers any improvement. 

For each possible branching at a node, C4.5 calculates the information gain ratio, while CART calculates the weighted Gini index to select the optimal attribute and threshold to branch the tree.

+ If an attribute is continuous, both algorithms bisect the data at a threshold.

+ If an attribute is categorical,
  
  + C4.5 groups the data by the attribute value and discard this attribute since. 
  
  + CART bisects the data by equal or not equal to a value for each distinct value of the attribute.

## Pruning

Both algorithms reject prepruning to maximize the predictive power of the model. However, to reduce time complexity, we may restrict the maximum depth of recursion, maximum number of nodes, etc when generating the tree.

The postpruning is done through a postorder traversal. Start from the bottom of the tree and examine each nonleaf subtree. If replacement of this subtree with a leaf, or with its most frequently used branch, would lead to a lower predicted error rate, then prune the tree accordingly. The two algorithms differ in how they predict these error rates.

In his book, Quinlan lists two common ways of estimating prediction errors:

+ Cost-complexity pruning [Breiman et al., 1984], in which the predicted error rate of a tree is modeled as the weighted sum of its complexity and its error on the training cases, with the separate cases used primarily to determine an appropriate weighting.

+ Reduced-error pruning [Quinlan, 1987e], which assesses the error rates of the tree and its components directly on the set of separate cases.

The drawback is simply that some of the available data must be reserved as validation set. One way around this problem is cross-validation. 

C4.5 and CART apply techniques as follows.

### C4.5

C4.5 uses a pruning technique called pessimistic pruning, which aims to avoid the necessity of a separate test data set. As has been seen, the misclassification rates produced by a tree on its training data are overly optimistic and, if used for pruning, produce overly large trees. Quinlan suggests using the continuity correction for the binomial distribution to obtain a more realistic estimate of the error rate.

Define $N(t) =$  number of training set examples at node t, $e(t) =$ number of example misclassified at node t. Then the error rate at t would be

$$r(t) = \frac{e(t)}{N(t)}$$

Apply the continuous correction, the error rate is estimated

$$r'(t) = \frac{e(t) + 1/2}{N(t)}$$

For a sub-tree $T_t$, whose root is node t, the corrected error rate is thus

$$r'(T_t) = \frac{\sum(e(i) + 1/2)}{\sum N(i)} = \frac{\sum e(i) + N_t/2}{N(t)}$$

where $N_t$ is the number of leaves in $T_t$. Since the expressions for $r'(t)$ and $r'(T_t)$ share the same denominator, we can instead calculate

$$n'(t) = e(t) + 1/2 \  \text{for a node}$$
$$n'(T_t) = \sum e(i) + N_T/2 \ \text{for a sub-tree}$$

With the training data, the sub-tree will always make fewer errors than the corresponding node. But this is not so if the corrected figures are used, since they also depend on the number of leaves. However, it is likely that even this corrected estimate of the number of misclassifications made by the sub-tree will be optimistic. So the algorithm only keeps the sub-tree if its corrected figure is more than one standard error better than the figure for the node.

The standard error is derived from binomial distribution $B(N(t), p)$, where $p = n'(T_t)/N(t)$.

$$ SE(n'(T_t)) = \sqrt{N(t) \times p \times (1-p)} = \sqrt{\frac{n'(T_t) \times (N(t) - n'(T_t))}{N(t)}}$$

The sub-tree $T_t$ is replaced with a single node $t$ if

$$ n'(T_t) - n'(t) > SE(n'(T_t)) $$

### CART

CART's method belongs to the first family

## Reference

1. J. Ross Quinlan. C4.5: programs for machine learning (1993).
   https://dl.acm.org/doi/book/10.5555/152181

2. Mingers, J. An Empirical Comparison of Pruning Methods for Decision Tree Induction. *Machine Learning*  **4,** 227–243 (1989).
   https://doi.org/10.1023/A:1022604100933

3. Steinberg, Dan A. Chapter 10 CART : Classification and Regression Trees (2009).
   https://www.taylorfrancis.com/books/e/9780429138423/chapters/10.1201/9781420089653-17

4. Decision tree pruning. Wikipedia.
   https://en.wikipedia.org/wiki/Decision_tree_pruning

## License

[MIT](https://choosealicense.com/licenses/mit/)
