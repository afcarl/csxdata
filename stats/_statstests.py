import numpy as np

from scipy.stats import f_oneway
from ..utilities.vectorop import split_by_categories


def manova(X: np.ndarray, Y: np.ndarray):
    categX = split_by_categories(Y, X)
    Xs = list(categX.values())
    F, p = f_oneway(*Xs)
    return F[0], p[0]
