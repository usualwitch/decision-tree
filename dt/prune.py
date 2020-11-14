import numpy as np

from .metrics import get_entropy, get_gini


def error(node):
    return (node.val[node.val != node.name]).shape[0]


def empirical_entropy(node):
    _, counts = np.unique(node.val, return_counts=True)
    return get_entropy(counts)


def gini_impurity(node):
    _, counts = np.unique(node.val, return_counts=True)
    return get_gini(counts)


def difference(node, loss_func=error):
    """Only use this function on a node with node.height == 1."""
    loss_as_node = loss_func(node)
    loss_as_subtree = sum(loss_func(sub_node) for sub_node in node.children)
    return loss_as_node - loss_as_subtree


def reduced_loss_prune(node, loss_func=error, **args):
    """Only use this function on a node with node.height == 1."""
    # If we discard node's branches, node.val will be classified to node.name class.
    if node.val.size == 0:
        return False
    if difference(node, loss_func) <= 0:
        node.children = ()
        return True
    return False


def pessimistic_prune(node, loss_func=error, **args):
    """
    Only use this function on a node with node.height == 1.

    Pessimistic pruning does not require an extra validation set.
    """
    if node.val.size == 0:
        node.children = ()
        return True
    n_examples = node.val.shape[0]
    loss_as_node = loss_func(node) + 1/2
    loss_as_subtree = sum(loss_func(sub_node) for sub_node in node.children) + len(node.children)/2
    if n_examples > loss_as_subtree:
        standard_loss = np.sqrt(loss_as_subtree*(n_examples-loss_as_subtree)/n_examples)
        if loss_as_subtree - loss_as_node > standard_loss:
            return False
    node.children = ()
    return True


def cost_complexity_prune(node, loss_func=error, alpha=0):
    if node.val.size == 0:
        node.children = ()
        return True
    loss_as_node = loss_func(node)/node.val.shape[0] + alpha
    loss_as_subtree = sum(loss_func(sub_node) for sub_node in node.children)/node.val.shape[0] + alpha*len(node.leaves)
    if loss_as_subtree <= loss_as_node:
        node.children = ()
        return True
    return False
