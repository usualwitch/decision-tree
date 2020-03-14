import pandas as pd

from tree import DecisionTree


df = pd.read_csv('data/balance_scale.csv')
dt = DecisionTree(df, algorithm='C4.5')
with open('dt2.txt', 'w') as f:
    print(dt, file=f)
r = dt.root
r1 = r.children[0]
r2 = r1.children[0]
