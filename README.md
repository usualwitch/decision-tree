# Decision Tree

An implementation of simplified C4.5 and CART algorithms.

A 10-fold CV is applied to evaluate the performance of algorithms.

## Usage

Clone the repo and run the code.

```
import pandas as pd
from sklearn.model_selection import train_test_split

from tree import DecisionTree
from preprocess import preprocess


df = pd.read_csv('data/mushrooms.csv')
df = preprocess(df)
train, test = train_test_split(df, test_size=0.33)
dt = DecisionTree(train, algorithm='C4.5', max_depth=100)
dt.predict(test)
```


## Algorithms

The algorithms are explained in [this post](https://blog.usualwitch.now.sh/posts/decision-tree/).

## License

[MIT](https://choosealicense.com/licenses/mit/)
