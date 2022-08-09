import numpy as np


def fillna_unknown_dtype_col(X):
    if X.dtype == np.int or X.dtype == np.int64:
        X = X.fillna(value=-1)
    elif X.dtype == np.float:
        X = X.fillna(value=-1)
    elif X.dtype == np.bool:
        X = X.fillna(value=False)
    elif X.dtype == np.object:
        X = X.fillna(value="missing")
    else:
        print(X.dtype)
    return X
