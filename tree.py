from anytree import Node, RenderTree, LevelOrderGroupIter
from anytree.render import ContRoundStyle
import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype
from sklearn.model_selection import train_test_split
import copy

from preprocess import preprocess
import metrics
import prune


class DecisionTree:

    ALGOS = {'C4.5', 'CART'}
    PRUNE_FUNCS = {'Reduce': prune.reduced_error_prune, 'Pessim': prune.pessimistic_error_prune}

    def __init__(self, data, algorithm='CART', prune_func=None, max_depth=10, ose_rule=False):
        """
        Supported algorithms: 'C4.5', 'CART'.

        CART does not support custom prune function. Supported prune functions for C4.5: 'Reduce', 'Pessim', 'Feature'.

        max_depth: positive int, the max depth of the decision tree.

        ose_rule: bool, whether to apply one standard error rule in CART's hyperparameter tuning.
        """
        # Configuration.
        if algorithm not in self.ALGOS:
            raise ValueError(f'The algorithm must be one of {self.ALGOS}.')
        self.algorithm = algorithm
        self.prune_func = prune_func
        self.max_depth = max_depth
        self.ose_rule = ose_rule

        if self.prune_func in self.PRUNE_FUNCS:
            if self.algorithm == 'C4.5':
                self.prune_func = prune_func
            else:
                raise ValueError('CART algorithm does not support other prune methods.')
        elif prune_func is None:
            if self.algorithm == 'C4.5':
                self.prune_func = 'Pessim'
        else:
            raise ValueError(f'The prune function must be one of {set(self.PRUNE_FUNCS)}.')

        # Converts the data types and marks target column.
        self.data = preprocess(data)

        # Define train/val sets.
        if self.algorithm == 'CART' or (self.algorithm == 'C4.5' and prune_func == 'Pessim'):
            train = self.data
            val = train
        else:
            train, val = train_test_split(self.data, test_size=0.33)

        # Generates the decision tree.
        self.root = self.generate_tree(train, val)

        # Prune the decision tree.
        self.postprune()

        # Print test accuracy.
        print(metrics.)

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
            return Node(target.mode()[0], attr='leaf', threshold='', val=val)
        # Keep only the valid attributes and the target column.
        train = train[valid_attrs + ['target']]

        # Recursion case.
        # Select optimal attribute.
        opt_attr, threshold = self.select_attr(train)

        # Create root node.
        root = Node(target.mode()[0], attr=opt_attr, threshold='', val=val)

        # Branching.
        # Delete the opt_attr from attr set only if C4.5 and categorical.
        if is_categorical_dtype(train[opt_attr].dtype) and self.algorithm == 'C4.5':
            for e in train[opt_attr].unique():
                branch_train = train[train[opt_attr] == e]
                branch_val = val[val[opt_attr] == e]
                if branch_train.empty:
                    # Generate a leaf node.
                    Node(target.mode()[0], parent=root, attr='leaf', threshold=e, val=branch_val)
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
                    Node(target.mode()[0], parent=root, attr='leaf', threshold=e, val=branch_val)
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

    def postprune(self):
        if self.algorithm == 'C4.5':
            # Postprune the tree in postorder traversal.
            self.postprune(self.root, self.PRUNE_FUNCS[self.prune_func])
        else:
            # Gets sorted alpha values by weakest link pruning.
            alpha_values = self.get_alphas()
            print(f'The alpha values are {alpha_values}.')
            # Selects best alpha by 10-fold CV.
            train_size = self.data.shape[0]*9//10
            val_size = self.data.shape[0] - train_size
            loss_table = {i: [] for i in range(len(alpha_values))}
            for k in range(10):
                val = self.data.iloc[k*val_size:(k+1)*val_size]
                train = self.data[~self.data.index.isin(val.index)]
                root_k = self.generate_tree(train, train)
                for i, alpha in enumerate(alpha_values):
                    tree_i = self.post_order_prune(root_k, lambda node: prune.cost_complexity_prune(node, alpha))
                    if tree_i:
                        loss_table[i].append(metrics.cost_complexity_loss(tree_i, val, alpha))
                    else:
                        loss_table[i].append(np.inf)
            loss_table = pd.DataFrame(loss_table)
            mean_loss = loss_table.mean()
            print(mean_loss)
            if self.ose_rule:
                se = mean_loss.std()
                opt_alpha = alpha_values[[i for i in range(len(alpha_values)) if mean_loss[i] <= mean_loss.min() + se][-1]]
            else:
                opt_alpha = alpha_values[[i for i in range(len(alpha_values)) if mean_loss[i] == mean_loss.min()][-1]]
            print(f'The selected alpha is {opt_alpha}')
            # Prune the tree using opt_alpha hyperparameter.
            self.post_order_prune(self.root, lambda node: prune.cost_complexity_prune(node, opt_alpha))

    def post_order_prune(self, node, prune_func):
        """Postprune node in postorder traversal. Only use this function on a node s.t. node.height >= 1."""
        pruned = False
        if node.height == 1:
            pruned = prune_func(node)
        elif node.height > 1:
            for child in node.children:
                if child.is_leaf:
                    continue
                # Recurse on a non-leaf child.
                self.post_order_prune(child, prune_func)
            if node.height == 1:
                pruned = prune_func(node)
        if pruned:
            node.attr = 'leaf'

    def get_alphas(self):
        """Creates a sequence of trees by weakest link pruning, and calculates alpha values."""
        # Create a deepcopy of self.root.
        root = copy.deepcopy(self.root)
        # Include all levels except leaves.
        levels = [[node for node in level] for level in LevelOrderGroupIter(root)][::-1][1:]
        alpha_values = [0]
        for level in levels:
            while level:
                node = level.pop()
                # Each time we prune a height==1 tree with 2 leaves.
                alpha = prune.error_difference(node)/2
                alpha_values.append(alpha)
                node.children = ()
        alpha_values = list(set(alpha_values))
        return sorted(alpha_values)

# TODO Test this function
    def predict(self, data):
        """
        Returns prediction results for data of the decision tree.

        If data contains valid target column, this function also prints out accuracy.
        """
        def rec_predict(data, node):
            # If node is a leaf, set prediction, jump out of recursion.
            if node.is_leaf:
                data['prediction'] = node.name
            # Recursion.
            if is_categorical_dtype(data[node.attr]) and self.algorithm == 'C4.5':
                for child in node.children:
                    branch_data = data[data[node.attr] == child.threshold]
                    rec_predict(branch_data, child)
            else:
                for child in node.children:
                    branch_data = data.query(f'{node.attr} {child.threshold}')
                    rec_predict(branch_data, child)

        labeled = 'target' in data.columns
        data = preprocess(data, labeled=labeled)
        rec_predict(data, self.root)

        if labeled:
            acc = (data['prediction'] == data['target']).sum()/data.shape[0]
            return acc, data['prediction']
        else:
            return data['prediction']
