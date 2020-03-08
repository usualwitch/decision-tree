from anytree import Node, RenderTree
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
        self.root = self.generate_tree(val=self.val)

    def generate_tree(self):
        """
        Generates a decision tree from training data, retaining validation set in each node for postpruning.

        data.columns = [attributes, target]. All attributes are transformed into categorical variables.
        """
        # Get attribute names, target and possible classes.
        attrs = self.train.columns[:-1]
        target = self.train.iloc[:, -1]
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
        opt_attr, partition_rules, is_categorical = self.select_attr()

        # Create root node.
        root = Node(opt_attr, val=self.val)

        # Branching.
        # if self.algorithm == 'C4.5':
        #     if is_categorical:
        #         for e in 
        # elif self.algorithm == 'CART':
        #     raise NotImplementedError


    def select_attr(self, criterion='gini'):
        """
        Selects optimal attribute for decision-tree branching.

        criterion = 'gini' or 'entropy'

        Returns:

        opt_attr: the attribute that yields the largest information gain

        partition_rules: a list of pandas query rules

        is_categorical: whether the selected attribute is categorical
        """
        assert criterion in {'gini', 'entropy'}, "criterion = 'gini' or 'entropy'"
        if criterion == 'entropy':
            

        return opt_attr, partition_rules, is_categorical


if __name__ == '__main__':
    df = pd.read_csv('data/balance_scale.csv')
    config = {'algorithm': 'C4.5', 'penalty_func': '', 'penalty_coeff': 1}
    dt = DecisionTree(df, config)
