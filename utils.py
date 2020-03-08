# TODO Penalty function
# TODO Memoization decorator for postorder postpruning

import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype, is_categorical_dtype


def preprocess(df):
    """
    Converts each column's dtype to either numeric (with > 10 unique values) or ordered categorical.

    The last column is the target, which must be categorical and named 'target'.

    Does nothing if this requirement is already satisfied.
    """
    for col in df.columns:
        col_dtype = df[col].dtype
        if is_numeric_dtype(col_dtype) and df[col].nunique() > 10:
            continue
        elif is_categorical_dtype(col_dtype) and df[col].cat.ordered:
            continue
        else:
            df[col] = df[col].astype('category')
        df[col] = df[col].cat.as_ordered()

    assert is_categorical_dtype(df.iloc[:, -1].dtype), 'Target must be categorical.'
    df.columns = [*df.columns[:-1], 'target']
    return df
