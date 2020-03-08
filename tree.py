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
        self.train, self.val = train_test_split(data, test_size=0.33, random_state=42)

        # Generates the decision tree.
        self.root = self.generate_tree()

    def generate_tree(self):
        """
        Generates a decision tree from training data, retaining validation set in each node for postpruning.

        data.columns = [attributes, target]. All attributes are transformed into categorical variables.
        """
        # Get attribute names, target and possible classes.
        attrs = self.train.columns[:-1]
        target = self.train['target']
        classes = target.cat.categories

        # Out of recursion cases:
        # Case 1: There is only one class.
        if len(classes) == 1:
            return Node(classes[0], val=self.val)
        # Case 2: There is no valid attribute, i.e. all values are the same for samples, or attrs is empty.
        valid_attrs = [a for a in attrs if self.train[a].nunique() > 1]
        if not valid_attrs:
            return Node(target.mode(), val=self.val)

        # Recursion case.
        # Select optimal attribute.
        opt_attr = self.select_attr()

        # Create root node.
        root = Node(opt_attr, val=self.val)

        # Branching.
        # if self.algorithm == 'C4.5':

        # elif self.algorithm == 'CART':
        #     raise NotImplementedError

    def select_attr(self):
        """
        Selects optimal attribute for decision-tree branching.
        """
        if self.algorithm == 'C4.5':
            pass
        else:
            pass

        opt_attr = None

        return opt_attr

    def evaluate_split(self, attr):
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
                # TODO binary split
                assert threshold is not None, 'Must provide threshold for continuous variables.'
                # bool attr

        df = self.train[[attr, 'target']]
        if self.algorithm == 'C4.5':
            entropy = get_entropy(df, 'target')
            cond_entropy = get_cond_entropy(df, attr)
            info_gain = entropy - cond_entropy
            intrinsic_value = get_entropy(df, attr)
            return info_gain / intrinsic_value
        else:
            



if __name__ == '__main__':
    df = pd.read_csv('data/balance_scale.csv')
    config = {'algorithm': 'C4.5', 'penalty_func': '', 'penalty_coeff': 1}
    dt = DecisionTree(df, config)
    dt.evaluate_split('left_weight')
