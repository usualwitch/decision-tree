import pandas as pd

from tree import DecisionTree


df = pd.read_csv('data/balance_scale.csv')
dt = DecisionTree(df, algorithm='CART')
with open('dt.txt', 'w') as f:
    print(dt, file=f)
