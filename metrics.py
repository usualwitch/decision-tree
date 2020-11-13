from pandas.api.types import is_categorical_dtype
import numpy as np

import prune


def get_counts(y):
    _, counts = np.unique(y, return_counts=True)
    return counts


def get_entropy(counts):
    proportions = counts / counts.sum()
    return - (proportions*np.log2(proportions)).sum()


def get_conditional_entropy(x, x_values, x_counts, y, threshold=None):
    if threshold is None:
        sub_entropies = np.array([get_entropy(get_counts(y[np.where(x == e)])) for e in x_values])
        proportions = x_counts / x_counts.sum()
        return (sub_entropies*proportions).sum()
    else:
        l_indices = np.where(x <= threshold)
        r_indices = np.where(x > threshold)
        l_entropy = get_entropy(get_counts(y[l_indices]))
        r_entropy = get_entropy(get_counts(y[r_indices]))
        l_count = l_indices[0].shape[0]
        r_count = r_indices[0].shape[0]
        return (l_entropy*l_count + r_entropy*r_count)/x.shape[0]
