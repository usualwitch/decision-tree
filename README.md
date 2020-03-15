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

Define <img src="/tex/7b115a9b8e26b2a91e0d6a0f5dd761e1.svg?invert_in_darkmode&sanitize=true" align=middle width=51.07303079999999pt height=24.65753399999998pt/>  number of training set examples at node t, <img src="/tex/e4ef0f4d82757ec8503701da9fe4abec.svg?invert_in_darkmode&sanitize=true" align=middle width=43.72720109999999pt height=24.65753399999998pt/> number of example misclassified at node t. Then the error rate at t would be

<p align="center"><img src="/tex/2265dbd8479bf7a5e48be3bda9c0fb53.svg?invert_in_darkmode&sanitize=true" align=middle width=95.4805071pt height=38.83491479999999pt/></p>

Apply the continuous correction, the error rate is estimated

<p align="center"><img src="/tex/7fc58a81bb1f3517b88ab1298b37ca4f.svg?invert_in_darkmode&sanitize=true" align=middle width=137.49536129999998pt height=38.83491479999999pt/></p>

For a sub-tree <img src="/tex/730cf633d28d7060a2f217a4cffe957c.svg?invert_in_darkmode&sanitize=true" align=middle width=14.57197829999999pt height=22.465723500000017pt/>, whose root is node t, the corrected error rate is thus

<p align="center"><img src="/tex/2cdfdd8bbe456c5416f4b8bf9f61ac89.svg?invert_in_darkmode&sanitize=true" align=middle width=303.39140864999996pt height=38.835134249999996pt/></p>

where <img src="/tex/56026e99b8e6524389672a202257d5aa.svg?invert_in_darkmode&sanitize=true" align=middle width=24.526322699999987pt height=24.65753399999998pt/> is the number of leaves in <img src="/tex/730cf633d28d7060a2f217a4cffe957c.svg?invert_in_darkmode&sanitize=true" align=middle width=14.57197829999999pt height=22.465723500000017pt/>. Since the expressions for <img src="/tex/87b707116f398e1ed56227e5bf16c4bb.svg?invert_in_darkmode&sanitize=true" align=middle width=35.94187574999999pt height=24.7161288pt/> and <img src="/tex/bfbde8695c9039fbf2794549c567d448.svg?invert_in_darkmode&sanitize=true" align=middle width=45.399652649999986pt height=24.7161288pt/> share the same denominator, we can instead calculate

<p align="center"><img src="/tex/6f7bf1d9ca91129a04d0ebc0cf42b103.svg?invert_in_darkmode&sanitize=true" align=middle width=209.39325329999997pt height=17.2895712pt/></p>
<p align="center"><img src="/tex/1b3bf4021cb591901bfd8ff9b6149801.svg?invert_in_darkmode&sanitize=true" align=middle width=287.5163016pt height=26.301595649999996pt/></p>

With the training data, the sub-tree will always make fewer errors than the corresponding node. But this is not so if the corrected figures are used, since they also depend on the number of leaves. However, it is likely that even this corrected estimate of the number of misclassifications made by the sub-tree will be optimistic. So the algorithm only keeps the sub-tree if its corrected figure is more than one standard error better than the figure for the node.

The standard error is derived from binomial distribution <img src="/tex/34cb995b028afb9f52e8684f6023b75c.svg?invert_in_darkmode&sanitize=true" align=middle width=75.37678664999999pt height=24.65753399999998pt/>, where <img src="/tex/4fad65502bced67933bff3bd2bc22120.svg?invert_in_darkmode&sanitize=true" align=middle width=113.8737204pt height=24.7161288pt/>.

<p align="center"><img src="/tex/a5cc7ebf9edade4e17f49f691d522a8c.svg?invert_in_darkmode&sanitize=true" align=middle width=463.75632269999994pt height=49.315569599999996pt/></p>

The sub-tree <img src="/tex/730cf633d28d7060a2f217a4cffe957c.svg?invert_in_darkmode&sanitize=true" align=middle width=14.57197829999999pt height=22.465723500000017pt/> is replaced with a single node <img src="/tex/4f4f4e395762a3af4575de74c019ebb5.svg?invert_in_darkmode&sanitize=true" align=middle width=5.936097749999991pt height=20.221802699999984pt/> if

<p align="center"><img src="/tex/67f23ca74b8e03a8d39352c0c4cc34e6.svg?invert_in_darkmode&sanitize=true" align=middle width=197.42021475pt height=17.2895712pt/></p>

### CART

CART's method belongs to the cost-complexity pruning family. Like C4.5, it only depends on the training data. The pruning algorithm aims to minimize the cost-complexity function
<p align="center"><img src="/tex/b9674fd35e041158d5388e18a498ead1.svg?invert_in_darkmode&sanitize=true" align=middle width=156.6817131pt height=16.438356pt/></p>
where <img src="/tex/d184151fc83468f7656df12762717093.svg?invert_in_darkmode&sanitize=true" align=middle width=37.28321519999999pt height=24.65753399999998pt/> and <img src="/tex/64d5b62235f6fbfef0ee9c4ec7f7624e.svg?invert_in_darkmode&sanitize=true" align=middle width=21.021758999999992pt height=24.65753399999998pt/> refer to tree <img src="/tex/2f118ee06d05f3c2d98361d9c30e38ce.svg?invert_in_darkmode&sanitize=true" align=middle width=11.889314249999991pt height=22.465723500000017pt/>'s error rate and number of leaves.

To solve the optimization problem, we use weakest link pruning:
+ Starting with the initial tree <img src="/tex/e3663010cccabb7ddb62cfc5391982b9.svg?invert_in_darkmode&sanitize=true" align=middle width=18.441856949999988pt height=26.76175259999998pt/>, substitute a sub-tree <img src="/tex/730cf633d28d7060a2f217a4cffe957c.svg?invert_in_darkmode&sanitize=true" align=middle width=14.57197829999999pt height=22.465723500000017pt/> with a leaf node <img src="/tex/4f4f4e395762a3af4575de74c019ebb5.svg?invert_in_darkmode&sanitize=true" align=middle width=5.936097749999991pt height=20.221802699999984pt/> to obtain <img src="/tex/3217e0efb21cd369bfd524269bcddd7d.svg?invert_in_darkmode&sanitize=true" align=middle width=18.441856949999988pt height=26.76175259999998pt/> by minimizing 
<p align="center"><img src="/tex/9d7293a2da3797c16a69bfa55b1160dd.svg?invert_in_darkmode&sanitize=true" align=middle width=234.01702335pt height=39.887022449999996pt/></p>
+ Iterate this pruning to obtain a sequence <img src="/tex/31b90387fe298a15828f8de6913f285b.svg?invert_in_darkmode&sanitize=true" align=middle width=105.9170013pt height=26.76175259999998pt/> where <img src="/tex/af89b1500e6871c84308dc0df267377f.svg?invert_in_darkmode&sanitize=true" align=middle width=23.55416084999999pt height=22.465723500000017pt/> is the null tree.
+ Select the optimal tree <img src="/tex/6f2d89667901be620b8e8faaf5f4fba6.svg?invert_in_darkmode&sanitize=true" align=middle width=16.54021049999999pt height=27.15900329999998pt/> by cross validation.

As shown [here](https://stats.stackexchange.com/questions/193538/how-to-choose-alpha-in-cost-complexity-pruning#), 
<p align="center"><img src="/tex/e464a0be1f7b49238b8aadca296309ed.svg?invert_in_darkmode&sanitize=true" align=middle width=362.1417657pt height=16.438356pt/></p>
Equate the above expression to <img src="/tex/29632a9bf827ce0200454dd32fc3be82.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=21.18721440000001pt/>, we get
<p align="center"><img src="/tex/6c3ff270d032cab33d9953aa0f62af80.svg?invert_in_darkmode&sanitize=true" align=middle width=133.21445279999998pt height=38.83491479999999pt/></p>

The tree sequence and the corresponding <img src="/tex/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode&sanitize=true" align=middle width=10.57650494999999pt height=14.15524440000002pt/>-values satisfy:
<p align="center"><img src="/tex/137a563f5bff74c1a783ca2a330bd2b4.svg?invert_in_darkmode&sanitize=true" align=middle width=152.4007221pt height=17.399144399999997pt/></p>
<p align="center"><img src="/tex/286088f4c86e142e27913fbd9a0c0f70.svg?invert_in_darkmode&sanitize=true" align=middle width=236.86772999999997pt height=16.43793525pt/></p>
 
<img src="/tex/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode&sanitize=true" align=middle width=10.57650494999999pt height=14.15524440000002pt/> is selected out of <img src="/tex/d4b05949cde15731185d926b75ce5427.svg?invert_in_darkmode&sanitize=true" align=middle width=93.98205464999998pt height=26.76175259999998pt/> by 10-fold cross validation:
+ Split the training points into 10 folds.
+ For <img src="/tex/d0b6aa1b351ee1eb5bcddf0383141ef8.svg?invert_in_darkmode&sanitize=true" align=middle width=92.18003684999998pt height=22.831056599999986pt/>, using every fold except the <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/>th:
  + Construct a sequence of trees <img src="/tex/31b90387fe298a15828f8de6913f285b.svg?invert_in_darkmode&sanitize=true" align=middle width=105.9170013pt height=26.76175259999998pt/> for <img src="/tex/328f2318a39b80cf9d881bbd5252af55.svg?invert_in_darkmode&sanitize=true" align=middle width=124.64969054999999pt height=26.76175259999998pt/>.
  + For each tree <img src="/tex/6f2d89667901be620b8e8faaf5f4fba6.svg?invert_in_darkmode&sanitize=true" align=middle width=16.54021049999999pt height=27.15900329999998pt/>, calculate the <img src="/tex/b40af3b549ee1504a2f3e52bdd3493c0.svg?invert_in_darkmode&sanitize=true" align=middle width=42.75601109999999pt height=27.15900329999998pt/> on the validation set, i.e. the <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/>th set.
+ Select parameter <img src="/tex/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode&sanitize=true" align=middle width=10.57650494999999pt height=14.15524440000002pt/> that minimizes the average the validation error.

## Reference

1. J. Ross Quinlan. C4.5: programs for machine learning (1993).
   https://dl.acm.org/doi/book/10.5555/152181

2. Mingers, J. An Empirical Comparison of Pruning Methods for Decision Tree Induction. Machine Learning, 4, 227–243 (1989).
   https://doi.org/10.1023/A:1022604100933

3. Hastie, T.; Tibshirani, R. & Friedman, J. (2001), The Elements of Statistical Learning, Springer New York Inc. , New York, NY, USA (2001).

## License

[MIT](https://choosealicense.com/licenses/mit/)
