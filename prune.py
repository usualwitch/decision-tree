def val_correct_cases(node):
    return node.val[node.val['target'] == node.name].shape[0]


def reduced_error(node):
    """Only use this function on a node s.t. node.height == 1."""
    # If we discard node's branches, node.val will be classified to node.name class.
    if node.val.empty:
        return False
    acc_prune = val_correct_cases(node)/node.val.shape[0]
    acc_no_prune = sum(val_correct_cases(sub_node) for sub_node in node.children)/node.val.shape[0]
    if acc_prune >= acc_no_prune:
        node.children = ()
        return True


def pessimistic_error(node):
    """
    Only use this function on a node s.t. node.height == 1.

    Pessimistic pruning does not require an extra validation set.
    """
    


def error_complexity(node):
    """Only use this function on a node s.t. node.height == 1."""
    pass
