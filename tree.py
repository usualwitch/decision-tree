from anytree import Node, RenderTree


class DecisionTree:
    def __init__(self, df):
        self.root = self.generate_tree(df)

    def generate_tree(self, df):
        """
        Generate decision tree from a dataframe df.
        df.columns = [attributes, target].
        """
        # Get attribute names, target and possible classes.
        attrs = df.columns[:-1]
        target = df.iloc[:, -1]
        classes = target.unique()

        # Out of recursion cases:
        # Case 1: There is only one class.
        if len(classes) == 1:
            return Node(classes[0])
        # Case 2: There is no valid attribute, i.e. all values are the same for samples, or attrs is empty.
        valid_attrs = [a for a in attrs if df[a].nunique() > 1]
        if not valid_attrs:
            return Node(target.mode())

        # Select optimal attribute.
        
        # Recursion.



for pre, fill, node in RenderTree(root):
    print("%s%s" % (pre, node.name))
