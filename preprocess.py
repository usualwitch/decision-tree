from pandas.api.types import is_numeric_dtype, is_categorical_dtype


def preprocess(df):
    """
    Converts dataframe's dtype and shuffle the dataframe.

    Each column's dtype is converted to either numeric (with > 10 unique values) or ordered categorical.

    The last column is the target, which must be categorical and named 'target'.
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

    if not is_categorical_dtype(df.iloc[:, -1].dtype):
        raise ValueError('Target must be categorical.')
    df.columns = [*df.columns[:-1], 'target']
    return df.sample(frac=1)