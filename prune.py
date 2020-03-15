import math


def error(node):
    return node.val[node.val['target'] != node.name].shape[0]


def reduced_error(node):
    """Only use this function on a node s.t. node.height == 1."""
    # If we discard node's branches, node.val will be classified to node.name class.
    if node.val.empty:
        return False
    error_as_node = error(node)
    error_as_subtree = sum(error(sub_node) for sub_node in node.children)
    if error_as_subtree >= error_as_node:
        node.children = ()
        return True
    return False


def pessimistic_error(node):
    """
    Only use this function on a node s.t. node.height == 1.

    Pessimistic pruning does not require an extra validation set.
    """
    if node.val.empty:
        node.children = ()
        return True
    num_of_examples = node.val.shape[0]
    error_as_node = error(node) + 1/2
    error_as_subtree = sum(error(sub_node) for sub_node in node.children) + len(node.children)/2
    if num_of_examples > error_as_subtree:
        standard_error = math.sqrt(error_as_subtree*(num_of_examples-error_as_subtree)/num_of_examples)
        if error_as_subtree - error_as_node > standard_error:
            return False
    node.children = ()
    return True


# def cost_complexity(node):
#     """Only use this function on a node s.t. node.height == 1."""
#     pass
