import operator
import copy
import numpy as np
from anytree import Node, RenderTree, LevelOrderGroupIter
from anytree.render import ContRoundStyle
from sklearn.model_selection import train_test_split

from .prune import *
from .metrics import get_entropy, get_conditional_entropy
from .utils import is_number


class DecisionTree:
    """
    C4.5 decision tree.
    """
    def __init__(self, X, y, prune_method='Comp', loss='entropy', max_depth=20, name='', random_state=0, output=True):
        """
        prune_method: 'Reduce', 'Pessim', 'Comp'.

        loss: 'entropy', 'gini', '0-1'.

        max_depth: positive int, the max depth of the decision tree.
        """
        # Configuration.
        self.max_depth = max_depth
        self.name = name
        self.output = output
        prune_func_dict = {
            'Reduce': reduced_loss_prune,
            'Pessim': pessimistic_prune,
            'Comp': cost_complexity_prune
        }
        loss_func_dict = {
            'entropy': empirical_entropy,
            'gini': gini_impurity,
            '0-1': error
        }
        self.prune_method = prune_method
        self.prune_func = prune_func_dict[prune_method]
        self.loss_func_str = loss
        self.loss_func = loss_func_dict[loss]

        # Define train/val sets.
        if self.prune_method == 'Pessim':
            X_train, y_train = X, y
            X_val, y_val = X, y
        else:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_state)

        # Generate the decision tree.
        self.root = self._generate_tree(X_train, y_train, X_val, y_val)
        # Output the tree.
        if self.output:
            with open(f'output/{self.name}_unpruned.txt', 'w') as f:
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
                    if '<=' in node.threshold or '>' in node.threshold:
                        tree_str += (pre + f'{node.threshold}->{node.name}\n')
                    else:
                        tree_str += (pre + f'={node.threshold}->{node.name}\n')
                else:
                    if '<=' in node.threshold or '>' in node.threshold:
                        tree_str += (pre + f'{node.threshold}: {node.feature}\n')
                    else:
                        tree_str += (pre + f'={node.threshold}: {node.feature}\n')
        return tree_str

    def _generate_tree(self, X_train, y_train, X_val, y_val, depth=0, removed_features=[]):
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
        valid_features = [i for i in range(X_train.shape[1]) if len(X_values[i]) > 1 and i not in removed_features]
        print('-'*20)
        print('valid features')
        print(valid_features)
        print('-'*20)
        if any((len(np.unique(y_train)) == 1, len(valid_features) == 0, depth >= self.max_depth)):
            return Node(mode, feature=None, threshold='', val=y_val)

        # Select the best feature. threshold = '' if the feature is categorical.
        best_feature, threshold = self._select_feature(X_train, y_train, X_values, X_counts, y_counts, valid_features)
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
                    branch = self._generate_tree(X_train_branch, y_train_branch, X_val_branch, y_val_branch, depth=depth+1, removed_features=removed_features)
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
                    removed_features.append(best_feature)
                    branch = self._generate_tree(X_train_branch, y_train_branch, X_val_branch, y_val_branch, depth=depth+1, removed_features=removed_features)
                    branch.parent = root
                    branch.threshold = e
        return root

    def _select_feature(self, X, y, X_values, X_counts, y_counts, valid_features):
        """
        Select the best feature for decision-tree branching.
        """
        # Store the scores for each feature and the threshold for binary split.
        # results's columns = 'feature', 'threshold', 'score'
        results = []
        entropy = get_entropy(y_counts)
        for i in valid_features:
            x = X[:, i]
            intrinsic_value = get_entropy(X_counts[i])
            if is_number(X[0, i]):
                x = x.astype('float')
                # Use the midpoints as thresholds.
                thresholds = (X_values[i][:-1] + X_values[i][1:])/2
                for threshold in thresholds:
                    conditional_entropy = get_conditional_entropy(x, X_values[i], X_counts[i], y, threshold)
                    info_gain = entropy - conditional_entropy
                    score = info_gain / intrinsic_value
                    results.append([i, threshold, score])
            else:
                conditional_entropy = get_conditional_entropy(x, X_values[i], X_counts[i], y)
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

    def _post_order_prune(self, node, prune_func, loss_func, alpha):
        """Postprune node in postorder traversal. Only use this function on a node with node.height >= 1."""
        if node.height == 1:
            prune_func(node, loss_func=loss_func, alpha=alpha)
        elif node.height > 1:
            for child in node.children:
                if child.is_leaf:
                    continue
                self._post_order_prune(child, prune_func, loss_func, alpha)
            if node.height == 1:
                prune_func(node, loss_func=loss_func, alpha=alpha)

    def postprune(self, alpha=0):
        """
        Postprune a tree using prune_method.
        """
        self._post_order_prune(self.root, self.prune_func, self.loss_func, alpha)
        if self.output:
            with open(f'output/{self.name}_{self.prune_method}_{alpha}_pruned.txt', 'w') as f:
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
                alpha = difference(node)/2
                node.children = ()
                if alpha > 0:
                    alpha_values.append(alpha)
        alpha_values = list(set(alpha_values))
        return sorted(alpha_values)

    def predict(self, X, root=None):
        """
        Return prediction results for data of the decision tree.
        """
        y_pred = np.zeros(X.shape[0])

        def predict_from_node(indices, node):
            # If node is a leaf, set prediction, jump out of recursion.
            if node.is_leaf:
                y_pred[indices] = int(node.name)
                return

            x = X[:, node.feature]
            if is_number(x[0]):
                x = x.astype('float')
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
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        acc = (y == y_pred).sum()/y.shape[0]
        print(f'The test accuracy is {acc*100:.2f}%')
        return acc
