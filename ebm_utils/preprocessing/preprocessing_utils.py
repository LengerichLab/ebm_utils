"""
Utilities for preprocessing data before passing it into EBM.
"""

import numpy as np


def fillna_unknown_dtype_col(col):
    """
        Fills in NA values in a column
        even when we don't know the dtype automatically.
    """
    if col.dtype in (np.int, np.int64):
        col = col.fillna(value=-1)
    elif col.dtype == np.float:
        col = col.fillna(value=-1)
    elif col.dtype == np.bool:
        col = col.fillna(value=False)
    elif col.dtype == np.object:
        col = col.fillna(value="missing")
    else:
        print(f"Can't fill NA for dtype {col.dtype}")
    return col
