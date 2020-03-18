from pandas.api.types import is_categorical_dtype
import numpy as np

import prune


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
