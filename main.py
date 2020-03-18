import pandas as pd
from sklearn.model_selection import train_test_split

from tree import DecisionTree
from preprocess import preprocess


df = pd.read_csv('data/titanic.csv')
df = preprocess(df)
train, test = train_test_split(df, test_size=0.33)
dt = DecisionTree(train, algorithm='CART', max_depth=10)
with open('dt.txt', 'w') as f:
    print(dt, file=f)

dt.predict(test)
