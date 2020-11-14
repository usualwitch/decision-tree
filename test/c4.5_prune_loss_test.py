import pandas as pd
from importlib import import_module
from sklearn.model_selection import train_test_split
from dt import DecisionTree


dataset_names = ['iris', 'german', 'page_blocks', 'seeds', 'wine']
results = []
columns = ['trial', 'name', 'prune_method', 'loss', 'acc']


for i in range(1, 6):
    for name in dataset_names:
        dataset = import_module('milksets.' + name)
        X_train, X_test, y_train, y_test = train_test_split(*dataset.load(), test_size=0.2, random_state=i)

        for prune_method in ['Reduce', 'Pessim', 'Comp']:
            for loss in ['entropy', 'gini', '0-1']:
                tr = DecisionTree(X_train, y_train, prune_method, loss, name=name)
                tr.postprune()  # Use alpha = 0 by default, since it's the best choice by experiment.
                acc = tr.score(X_test, y_test)
                results.append([i, name, prune_method, loss, acc])


df = pd.DataFrame(results, columns=columns)
df.to_csv('prune_loss_test.csv', index=False)
