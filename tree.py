import operator
import copy
import numpy as np
from anytree import Node, RenderTree, LevelOrderGroupIter
from anytree.render import ContRoundStyle
from sklearn.model_selection import train_test_split

import metrics
import prune
from utils import is_number


class DecisionTree:
    """
    C4.5 decision tree.
    """
    def __init__(self, X, y, prune_method='Comp', max_depth=100, name=''):
        """
        prune_method: 'Reduce', 'Pessim', 'Comp', 'CompSqr'.

        max_depth: positive int, the max depth of the decision tree.
        """
        # Configuration.
        self.max_depth = max_depth
        prune_func_dict = {'Reduce': prune.reduced_error_prune,
                           'Pessim': prune.pessimistic_error_prune,
                           'Comp': lambda x: prune.cost_complexity_prune(x, 0.1),
                           'CompSqr': lambda x: prune.cost_complexity_squared_prune(x, 0.01)}
        if prune_method not in prune_func_dict:
            raise ValueError('Prune method is invalid.')
        self.prune_method = prune_method
        self.prune_func = prune_func_dict[prune_method]

        # Define train/val sets.
        if self.prune_method == 'Pessim':
            X_train, y_train = X, y
            X_val, y_val = X, y
        else:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

        # Generate the decision tree.
        self.root = self._generate_tree(X_train, y_train, X_val, y_val)
        with open(f'output/{name}_unpruned.txt', 'w') as f:
            print(self, file=f)

    def __str__(self):
        tree_str = ''
        for i, (pre, _, node) in enumerate(RenderTree(self.root, style=ContRoundStyle())):
            if i == 0:
                if self.root.is_leaf:
                    # Only one node in the tree.
                    tree_str += str(self.root.name)
                else:
                    tree_str += (str(self.root.feature) + '\n')
            else:
                if node.is_leaf:
                    if '<=' in node.threshold or '>' in threshold:
                        tree_str += (pre + f'{node.threshold}->{node.name}\n')
                    else:
                        tree_str += (pre + f'={node.threshold}->{node.name}\n')
                else:
                    if '<=' in node.threshold or '>' in threshold:
                        tree_str += (pre + f'{node.threshold}: {node.feature}\n')
                    else:
                        tree_str += (pre + f'={node.threshold}: {node.feature}\n')
        return tree_str

    def _generate_tree(self, X_train, y_train, X_val, y_val, depth=0):
        """
        Generate a decision tree from training data, retaining validation set in each node for postpruning.

        Tree structure:

            leaf node: Node(classified_class, feature=None, threshold, val)

            non-leaf node: Node(majority_class, feature=best_feature, threshold, val)
        """

        X_train = np.copy(X_train)
        y_train = np.copy(y_train)
        X_val = np.copy(X_val)
        y_val = np.copy(y_val)

        X_values = []
        X_counts = []
        for i in range(X_train.shape[1]):
            Xi_values, Xi_counts = np.unique(X_train[:, i], return_counts=True)
            if is_number(Xi_values[0]):
                Xi_values = Xi_values.astype('float')
            X_values.append(Xi_values)
            X_counts.append(Xi_counts)

        y_values, y_counts = np.unique(y_train, return_counts=True)
        mode = y_values[np.argmax(y_counts)]

        # Out of recursion cases return a leaf node.
        # 1. There is only one class.
        # 2. There is no valid feature, i.e. all values are the same for samples, or the left feature set is empty.
        # 3. Maximum tree depth is reached.
        valid_features = [i for i in range(X_train.shape[1]) if len(X_values[i]) > 1]
        print('-'*20)
        print('valid features')
        print(valid_features)
        print('-'*20)
        if any((len(np.unique(y_train)) == 1, not valid_features, depth >= self.max_depth)):
            return Node(mode, feature=None, threshold='', val=y_val)

        # Keep only the valid features.
        X_train = X_train[:, valid_features]

        # Select best feature. threshold = '' if the feature is categorical.
        best_feature, threshold = self._select_feature(X_train, y_train, X_values, X_counts, y_counts)
        print('best_feature')
        print(best_feature)
        print('-'*20)
        root = Node(mode, feature=best_feature, threshold=threshold, val=y_val)

        # Branching.
        x = X_train[:, best_feature]
        x_val = X_val[:, best_feature]
        if is_number(x[0]):
            x = x.astype('float')
            x_val = x_val.astype('float')
            binary_dict = {'<=': operator.__le__, '>': operator.__gt__}
            for name, operation in binary_dict.items():
                train_indices = np.where(operation(x, threshold))
                X_train_branch = X_train[train_indices]
                y_train_branch = y_train[train_indices]
                val_indices = np.where(operation(x_val, threshold))
                X_val_branch = X_val[val_indices]
                y_val_branch = y_val[val_indices]
                if X_train_branch.size == 0:
                    # Generate a leaf node that inherits its parent value.
                    Node(mode, parent=root, feature=None, threshold=f'{name}{threshold}', val=y_val_branch)
                else:
                    branch = self._generate_tree(X_train_branch, y_train_branch, X_val_branch, y_val_branch, depth=depth+1)
                    branch.parent = root
                    branch.threshold = f'{name}{threshold}'
        else:
            for e in X_values[best_feature]:
                train_indices = np.where(x == e)
                X_train_branch = X_train[train_indices]
                y_train_branch = y_train[train_indices]
                val_indices = np.where(x_val == e)
                X_val_branch = X_val[val_indices]
                y_val_branch = y_val[val_indices]
                if X_train_branch.size == 0:
                    # Generate a leaf node that inherits its parent value.
                    Node(mode, parent=root, feature=None, threshold=e, val=y_val_branch)
                else:
                    # Remove the column of categorical best feature.
                    X_train_branch = np.delete(X_train_branch, best_feature, axis=1)
                    X_val_branch = np.delete(X_val_branch, best_feature, axis=1)
                    branch = self._generate_tree(X_train_branch, y_train_branch, X_val_branch, y_val_branch, depth=depth+1)
                    branch.parent = root
                    branch.threshold = e
        return root

    def _select_feature(self, X, y, X_values, X_counts, y_counts):
        """
        Select the best feature for decision-tree branching.
        """
        # Store the scores for each feature and the threshold for binary split.
        # results's columns = 'feature', 'threshold', 'score'
        results = []
        entropy = metrics.get_entropy(y_counts)
        for i in range(X.shape[1]):
            x = X[:, i]
            intrinsic_value = metrics.get_entropy(X_counts[i])
            if is_number(X[0, i]):
                x = x.astype('float')
                # Use the midpoints as thresholds.
                thresholds = (X_values[i][:-1] + X_values[i][1:])/2
                for threshold in thresholds:
                    conditional_entropy = metrics.get_conditional_entropy(x, X_values[i], X_counts[i], y, threshold)
                    info_gain = entropy - conditional_entropy
                    score = info_gain / intrinsic_value
                    results.append([i, threshold, score])
            else:
                conditional_entropy = metrics.get_conditional_entropy(x, X_values[i], X_counts[i], y)
                info_gain = entropy - conditional_entropy
                score = info_gain / intrinsic_value
                results.append([i, np.nan, score])

        results = np.array(results)
        best_index = np.argmax(results[:, 2])
        best_feature, threshold = results[best_index, :2]
        best_feature = int(best_feature)
        if threshold == np.nan:
            threshold = ''
        return best_feature, threshold

    def _post_order_prune(self, node, prune_func):
        """Postprune node in postorder traversal. Only use this function on a node with node.height >= 1."""
        pruned = False
        if node.height == 1:
            pruned = prune_func(node)
        elif node.height > 1:
            for child in node.children:
                if child.is_leaf:
                    continue
                self._post_order_prune(child, prune_func)
            if node.height == 1:
                pruned = prune_func(node)

    def postprune(self):
        """
        Postprune a tree using prune_method.
        """
        self._post_order_prune(self.root, self.prune_func)
        with open(f'output/{name}_pruned.txt', 'w') as f:
            print(self, file=f)

    def get_alphas(self):
        """
        Create a sequence of trees by weakest link pruning, and calculate alpha values.
        """
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

    def predict(self, X, y=None, root=None):
        """
        Return prediction results for data of the decision tree.
        """
        y_pred = np.zeros([X.shape[0], 1])

        def predict_from_node(indices, node):
            # If node is a leaf, set prediction, jump out of recursion.
            if node.is_leaf:
                y_pred[indices] = node.name
                return

            x = X[0, node.feature]
            if is_number(x[0]):
                for child in node.children:
                    operation = operator.__gt__ if child.threshold.startswith('>') else operator.__le__
                    threshold = float(child.threshold.strip('><='))
                    branch_indices = np.where(operation(x, threshold))
                    predict_from_node(branch_indices, child)
            else:
                for child in node.children:
                    branch_indices = np.where(x == child.threshold)
                    predict_from_node(branch_indices, child)

        if root is None:
            root = self.root
        predict_from_node((np.arange(X.shape[0]), ), root)

        if y is None:
            return y_pred
        else:
            acc = (y == y_pred).sum()/y.shape[0]
            print(f'The test accuracy is {acc*100:.2f}%')
            return y_pred, acc
