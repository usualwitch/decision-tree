from anytree import Node, RenderTree
from anytree.render import ContRoundStyle
import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype
from sklearn.model_selection import train_test_split

from preprocess import preprocess
import metrics
import prune


class DecisionTree:

    ALGOS = {'C4.5', 'CART'}
    PRUNE_FUNCS = {'Reduce': prune.reduced_error, 'Pessim': prune.pessimistic_error, 'Err-comp': prune.error_complexity}

    def __init__(self, data, algorithm='CART', prune_func=None, max_depth=100):
        """Supported algorithms: 'C4.5', 'CART'."""
        # Configuration.
        if algorithm not in self.ALGOS:
            raise ValueError(f'The algorithm must be one of {self.ALGOS}.')
        self.algorithm = algorithm
        self.max_depth = max_depth

        if prune_func is None:
            if self.algorithm == 'C4.5':
                self.prune_func = 'Pessim'
            else:
                self.prune_func = 'Err-comp'
        elif prune_func in self.PRUNE_FUNCS:
            self.prune_func = prune_func
        else:
            raise ValueError(f'The prune function must be one of {set(self.PRUNE_FUNCS)}.')

        # Converts and shuffles the data.
        data = preprocess(data)
        # # Cross validation.
        # train_size = data.shape[0]*9//10
        # val_size = data.shape[0] - train_size

        # for i in range(10):
        #     val = data.iloc[i*val_size:(i+1)*val_size]
        #     train = pd.concat([data.iloc[:i*val_size], data.iloc[(i+1)*val_size]])
        if self.algorithm == 'C4.5' and prune_func == 'Pessim':
            train = data
            val = pd.DataFrame()
        else:
            train, val = train_test_split(data, test_size=0.33)

        # Generates the decision tree.
        self.root = self.generate_tree(train, val)

        # Postprune the tree in postorder traversal.
        self.postprune(self.root, self.PRUNE_FUNCS[self.prune_func])

    def __str__(self):
        tree_str = ''
        for i, (pre, _, node) in enumerate(RenderTree(self.root, style=ContRoundStyle())):
            if i == 0:
                tree_str += (self.root.attr + '\n')
            else:
                if node.attr == 'leaf':
                    tree_str += (pre + f'{node.threshold}->{node.name}\n')
                else:
                    tree_str += (pre + f'{node.threshold}->{node.attr}\n')
        return tree_str

    def generate_tree(self, train, val, depth=0):
        """
        Generates a decision tree from training data, retaining validation set in each node for postpruning.

        Tree structure:

            leaf node: Node(classified_class, attr='leaf', threshold, val)

            non-leaf node: Node(majority_class, attr=opt_attr, threshold, val)

        depth helps record recursion parameters.
        """
        # Get attribute names, target and possible classes.
        target = train['target']

        """
        Out of recursion cases return a leaf node.
        Case 1: There is only one class.
        Case 2: There is no valid attribute, i.e. all values are the same (if binary, the boolean of attr >= threshold) for samples, or attrs is empty.
        Case 3: Maximum tree depth is reached.
        """
        valid_attrs = [a for a in train.columns if a != 'target' and train[a].nunique() > 1]
        if target.nunique() == 1 or not valid_attrs or depth >= self.max_depth:
            return Node(target.mode()[0], attr='leaf', threshold='', train=train, val=val)
        # Keep only the valid attributes and the target column.
        train = train[valid_attrs + ['target']]

        # Recursion case.
        # Select optimal attribute.
        opt_attr, threshold = self.select_attr(train)

        # Create root node.
        root = Node(target.mode()[0], attr=opt_attr, threshold='', train=train, val=val)

        # Branching.
        # Delete the opt_attr from attr set only if C4.5 and categorical.
        if is_categorical_dtype(train[opt_attr].dtype) and self.algorithm == 'C4.5':
            for e in train[opt_attr].unique():
                branch_train = train[train[opt_attr] == e]
                branch_val = val[val[opt_attr] == e]
                if branch_train.empty:
                    # Generate a leaf node.
                    Node(target.mode()[0], parent=root, attr='leaf', threshold=e, train=branch_train, val=branch_val)
                else:
                    branch_train = branch_train.drop(columns=[opt_attr])
                    branch_val = branch_val.drop(columns=[opt_attr])
                    branch = self.generate_tree(branch_train, branch_val, depth=depth+1)
                    branch.parent = root
                    branch.threshold = e
        else:
            for e in ['>=', '<']:
                branch_train = train.query(f'{opt_attr} {e} {threshold}')
                branch_val = val.query(f'{opt_attr} {e} {threshold}')
                if branch_train.empty:
                    # Generate a leaf node.
                    Node(target.mode()[0], parent=root, attr='leaf', threshold=e, train=branch_train, val=branch_val)
                else:
                    branch = self.generate_tree(branch_train, branch_val, depth=depth+1)
                    branch.parent = root
                    branch.threshold = f'{e}{threshold}'
        return root

    def select_attr(self, train):
        """
        Selects optimal attribute for decision-tree branching.
        """
        # Store the scores for each attribute and the threshold for binary split.
        result = {'attr_name': [], 'score': [], 'threshold': []}
        for attr in train.columns[:-1]:
            result['attr_name'].append(attr)
            if is_categorical_dtype(train[attr].dtype):
                if self.algorithm == 'C4.5':
                    result['score'].append(self.evaluate_split(train, attr))
                    result['threshold'].append(np.nan)
                    continue
                else:
                    cut_points = train[attr].unique().sort_values()[1:]
            else:
                # Try using the midpoints of continous variable, all but the smallest point of discrete variable as threshold.
                points = np.unique(np.sort(train[attr].values))
                cut_points = (points[:-1] + points[1:])/2
            sub_scores = pd.DataFrame(cut_points, columns=['threshold'])
            sub_scores['score'] = sub_scores['threshold'].apply(lambda x: self.evaluate_split(train, attr, threshold=x))
            sub_opt = sub_scores.iloc[sub_scores['score'].idxmax()]
            result['score'].append(sub_opt['score'])
            result['threshold'].append(sub_opt['threshold'])

        result = pd.DataFrame(result)
        opt = result.iloc[result['score'].idxmax()]
        # opt['threshold'] is np.nan if C4.5 and categorical.
        return opt['attr_name'], opt['threshold']

    def evaluate_split(self, train, attr, threshold=None):
        """
        Returns information gain ratio in C4.5.

        Returns weighted Gini index in CART.
        """
        df = train[[attr, 'target']]
        if self.algorithm == 'C4.5':
            entropy = metrics.get_entropy(df, 'target')
            cond_entropy = metrics.get_cond_entropy(df, attr, threshold)
            info_gain = entropy - cond_entropy
            intrinsic_value = metrics.get_entropy(df, attr)
            return info_gain / intrinsic_value
        else:
            return metrics.get_cond_gini(df, attr, threshold)

    def postprune(self, node, prune_func):
        """Postprune self.root in postorder traversal. Only use this function on a node s.t. node.height >= 1."""
        pruned = False
        if node.height == 1:
            pruned = prune_func(node)
        elif node.height > 1:
            for child in node.children:
                if child.is_leaf:
                    continue
                # Recurse on a non-leaf child.
                self.postprune(child, prune_func)
            if node.height == 1:
                pruned = prune_func(node)
        if pruned:
            node.attr = 'leaf'
