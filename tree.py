from anytree import Node, RenderTree
import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype
from sklearn.model_selection import train_test_split

from utils import preprocess


class DecisionTree:

    ALGO_SET = {'C4.5', 'CART'}
    PENALTY_SET = {''}

    def __init__(self, data, config={'algorithm': 'CART'}):
        """
        Supported algorithms: 'C4.5', 'CART'

        Supported penalty functions: TODO

        The penalty coefficient must be greater than or equal to 0.
        """
        # Configuration.
        self.algorithm = config['algorithm']
        self.penalty_func = config['penalty_func']
        self.penalty_coeff = config['penalty_coeff']
        if self.algorithm not in self.ALGO_SET or self.penalty_func not in self.PENALTY_SET:
            raise ValueError(f'The algorithm and the penalty function must be selected out of {self.ALGO_SET} and {self.PENALTY_SET}.')
        if self.penalty_coeff <= 0:
            raise ValueError('The penalty coefficient must be greater than or equal to 0.')

        # Preprocesses the data.
        data = preprocess(data)
        train, val = train_test_split(data, test_size=0.33, random_state=42)

        # Generates the decision tree.
        self.root = self.generate_tree(train, val)

        # Postprune the tree in postorder traversal.
        self.postprune(self.root)

    def generate_tree(self, train, val):
        """
        Generates a decision tree from training data, retaining validation set in each node for postpruning.
        Tree structure:
            leaf node: Node(classified_class, attr='leaf', threshold, val)
            non-leaf node: Node(majority_class, attr=opt_attr, threshold, val)
        """
        # Get attribute names, target and possible classes.
        attrs = train.columns[:-1]
        target = train['target']
        classes = target.cat.categories

        # Out of recursion cases return a leaf node.
        # Case 1: There is only one class.
        if len(classes) == 1:
            return Node(classes[0], attr='leaf', val=val)
        # Case 2: There is no valid attribute, i.e. all values are the same for samples, or attrs is empty.
        valid_attrs = [a for a in attrs if train[a].nunique() > 1]
        if not valid_attrs:
            return Node(target.mode()[0], attr='leaf', val=val)
        # Keep only the valid attributes and the target column.
        train = train[valid_attrs + ['target']]

        # Recursion case.
        # Select optimal attribute.
        opt_attr, threshold = self.select_attr(train)

        # Create root node.
        root = Node(target.mode()[0], attr=opt_attr, val=val)

        # Branching.
        # Delete the opt_attr from attr set only if C4.5 and categorical.
        if is_categorical_dtype(train[opt_attr].dtype) and self.algorithm == 'C4.5':
            for e in train[opt_attr].cat.categories:
                branch_train = train[train[opt_attr] == e]
                branch_val = val[val[opt_attr] == e]
                if branch_train.empty:
                    # Generate a leaf node.
                    Node(target.mode()[0], parent=root, attr='leaf', threshold=e, val=branch_val)
                else:
                    branch_train = branch_train.drop(columns=[opt_attr])
                    branch_val = branch_val.drop(columns=[opt_attr])
                    branch = self.generate_tree(branch_train, branch_val)
                    branch.parent = root
                    branch.threshold = e
        else:
            for e in ['>=', '<']:
                branch_train = train.query(f'{opt_attr} {e} {threshold}')
                branch_val = val.query(f'{opt_attr} {e} {threshold}')
                if branch_train.empty:
                    Node(target.mode()[0], parent=root, val=branch_val)
                else:
                    branch = self.generate_tree(branch_train, branch_val)
                    branch.parent = root
                    branch.threshold = f'{e} {threshold}'
        return root

    def select_attr(self, train):
        """
        Selects optimal attribute for decision-tree branching.
        """
        # Store the scores for each attribute and the threshold for binary split
        result = {'attr_name': [], 'score': [], 'threshold': []}
        for attr in train.columns[:-1]:
            result['attr_name'].append(attr)
            if is_categorical_dtype(train[attr].dtype):
                if self.algorithm == 'C4.5':
                    result['score'].append(self.evaluate_split(train, attr))
                    result['threshold'].append(np.nan)
                    continue
                else:
                    # The categorical variable is sorted.
                    cut_points = train[attr].cat.categories[1:]
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
        def get_proportion(df, attr):
            return df[attr].value_counts()/df[attr].shape[0]

        def get_entropy(df, attr):
            proportion = get_proportion(df, attr)
            return - (proportion*np.log2(proportion)).sum()

        def get_cond_entropy(df, attr, threshold=None):
            if is_categorical_dtype(df[attr].dtype):
                sub_entropies = df.groupby(attr).apply(lambda df: get_entropy(df, 'target'))
                proportion = get_proportion(df, attr)
                return (sub_entropies*proportion).sum()
            else:
                if threshold is None:
                    raise ValueError('Must provide threshold for continuous variables.')
                r_part = df[df[attr] >= threshold]
                l_part = df[df[attr] < threshold]
                r_entropy = get_entropy(r_part, 'target')
                l_entropy = get_entropy(l_part, 'target')
                return (r_entropy*r_part.shape[0] + l_entropy*l_part.shape[0])/(r_part.shape[0] + l_part.shape[0])

        def get_gini(df, attr):
            proportion = get_proportion(df, attr)
            return 1 - (proportion**2).sum()

        def get_cond_gini(df, attr, threshold):
            """The data is divided into >= threshold part and < threshold part."""
            r_part = df[df[attr] >= threshold]
            l_part = df[df[attr] < threshold]
            r_gini = get_gini(r_part, 'target')
            l_gini = get_gini(l_part, 'target')
            return (r_gini*r_part.shape[0] + l_gini*l_part.shape[0])/(r_part.shape[0] + l_part.shape[0])

        df = train[[attr, 'target']]
        if self.algorithm == 'C4.5':
            entropy = get_entropy(df, 'target')
            cond_entropy = get_cond_entropy(df, attr, threshold)
            info_gain = entropy - cond_entropy
            intrinsic_value = get_entropy(df, attr)
            return info_gain / intrinsic_value
        else:
            return get_cond_gini(df, attr, threshold)

    def postprune(self, node):
        """Postprune self.root in postorder traversal. Only use this function on a node s.t. node.height >= 1."""

        def prune(node):
            """Only use this function on a node s.t. node.height == 1."""
            def count_correct_cases(node):
                return node.val[node.val['target'] == node.name].shape[0]

            # If we discard node's branches, node.val will be classified to node.name class.
            count_prune = count_correct_cases(node)
            count_no_prune = sum(count_correct_cases(sub_node) for sub_node in node.children)
            if count_prune >= count_no_prune:
                node.children = ()

        if node.height == 1:
            prune(node)
        elif node.height > 1:
            for child in node.children:
                if child.is_leaf:
                    continue
                # Recurse on a non-leaf child.
                self.postprune(child)
            if node.height == 1:
                prune(node)


if __name__ == '__main__':
    df = pd.read_csv('data/knowledge.csv')
    config = {'algorithm': 'C4.5', 'penalty_func': '', 'penalty_coeff': 1}
    dt = DecisionTree(df, config)
    for pre, fill, node in RenderTree(dt.root):
        print("%s%s" % (pre, node.attr))
