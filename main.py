import pandas as pd

from tree import DecisionTree


df = pd.read_csv('data/balance_scale.csv')
dt = DecisionTree(df, algorithm='C4.5')
with open('dt.txt', 'w') as f:
    print(dt, file=f)
