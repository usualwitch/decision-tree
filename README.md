# Decision Tree

An implementation of simplified C4.5 and CART algorithms.

A 10-fold CV is applied to evaluate the performance of algorithms.

## Usage

Clone the repo and run the code.

```python
import pandas as pd
from tree import DecisionTree


df = pd.read_csv('data/balance_scale.csv')
dt = DecisionTree(df, algorithm='C4.5')
with open('dt.txt', 'w') as f:
    print(dt, file=f)
```

## Algorithms

The algorithms are explained in [this post](https://blog.usualwitch.now.sh/posts/decision-tree/).

## License

[MIT](https://choosealicense.com/licenses/mit/)
