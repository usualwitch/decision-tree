import pandas as pd

from tree import DecisionTree


df = pd.read_csv('data/balance_scale.csv')
config = {'algorithm': 'C4.5', 'penalty_func': '', 'penalty_coeff': 1}
dt = DecisionTree(df, config)
