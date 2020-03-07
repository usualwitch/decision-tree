from anytree import Node, RenderTree
import pandas as pd


class DecisionTree:

    ALGO_SET = {'C4.5', 'CART'}

    def __init__(self, train, val, algorithm='CART'):
        self.algorithm = algorithm
        if algorithm not in self.ALGO_SET:
            raise ValueError(f'The algorithm must be one of {self.ALGO_SET}.')
        self.train = train
        self.val = val
        self.root = self.generate_tree()

    def generate_tree(self):
        """
        Generate a decision tree from training data.
        data.columns = [attributes, target].
        """
        # Get attribute names, target and possible classes.
        attrs = self.train.columns[:-1]
        target = self.train.iloc[:, -1]
        classes = target.unique()

        # Out of recursion cases:
        # Case 1: There is only one class.
        if len(classes) == 1:
            return Node(classes[0])
        # Case 2: There is no valid attribute, i.e. all values are the same for samples, or attrs is empty.
        valid_attrs = [a for a in attrs if self.train[a].nunique() > 1]
        if not valid_attrs:
            return Node(target.mode())

        # Recursion case.
        # Select optimal attribute.
        opt_attr, attr_values = self.select_attr()

        # Create root node.
        if self.algorithm == 'C4.5':
            root = Node(opt_attr, attr_values)
        elif self.algorithm == 'CART':
            raise NotImplementedError
        # Branching.
        for e in attr_values:
            


    def select_attr(self):
        """
        Select optimal attribute for decision-tree branching.
        """
        return self.train.columns[0], self.train.iloc[:, 0].unique()


if __name__ == '__main__':
    df = pd.read_csv('data/knowledge.csv')
    dt = DecisionTree(df, 0, algorithm='C4.5')
