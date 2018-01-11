import warnings

import numpy as np
import pandas as pd


def asmatrix(X, getnames=False, matrixwarn=True):
    names = None
    if isinstance(X, pd.DataFrame):
        names = X.columns.tolist()
        X = X.as_matrix()
    if X.ndim > 2:
        if matrixwarn:
            warnings.warn("Raveling multidimensional data to matrix!", RuntimeWarning)
        X = X.reshape(len(X), np.prod(X.shape[1:]))
    return (X, names) if getnames else X
