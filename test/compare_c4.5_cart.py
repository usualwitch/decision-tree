import numpy as np
import pandas as pd
from importlib import import_module
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from dt import DecisionTree


dataset_names = ['iris', 'german', 'page_blocks', 'seeds', 'wine']
results = []
columns = ['trial', 'name', 'tree', 'alpha', 'acc']


for i in range(1, 2):
    for name in dataset_names:
        dataset = import_module('milksets.' + name)
        X_train, X_test, y_train, y_test = train_test_split(*dataset.load(), test_size=0.2, random_state=i)

        # C4.5 without pruning. Calculate this first to get alphas.
        tr = DecisionTree(X_train, y_train, loss='entropy', name=name)
        alphas = tr.get_alphas()
        accs = np.zeros([5, len(alphas)])

        # Select best alpha value using 5-fold stratifiled cv.
        skf = StratifiedKFold(n_splits=5, random_state=i)
        for j, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
            tr = DecisionTree(X_train[train_index], y_train[train_index], output=False)
            for k, alpha in enumerate(alphas):
                tr.postprune(alpha)
                acc = tr.score(X_test, y_test)
                accs[j, k] = acc
        accs = accs.mean(axis=0)
        best_alpha_index = accs.argmax()
        best_alpha = alphas[best_alpha_index]

        # Train and prune with best alpha.
        tr = DecisionTree(X_train, y_train, loss='entropy', name=name)
        tr.postprune(best_alpha)
        acc = tr.score(X_test, y_test)
        results.append([i, name, 'C4.5', best_alpha, acc])

        # scikit-learn decision tree grid search with alphas.
        parameters = {'ccp_alpha': alphas}
        sktr = DecisionTreeClassifier(criterion='entropy')
        grid_search = GridSearchCV(sktr, parameters, cv=5)
        grid_search.fit(X_train, y_train)
        acc = grid_search.best_estimator_.score(X_test, y_test)
        best_alpha = grid_search.best_params_['ccp_alpha']
        results.append([i, name, 'CART', best_alpha, acc])  # Use alpha == -2 to represent sklearn tree result.

df = pd.DataFrame(results, columns=columns)
df.to_csv('c4.5_cart_results.csv', index=False)
