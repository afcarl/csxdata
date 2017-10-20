import numpy as np

from scipy.stats import f_oneway
from ..utilities.vectorop import split_by_categories


def manova(X: np.ndarray, Y: np.ndarray):
    categX = split_by_categories(Y, X)
    Xs = list(categX.values())
    F, p = f_oneway(*Xs)
    return F[0], p[0]


def _simple_T2(X, means, cov):
    from scipy import stats
    N, dim = X.shape
    Xbar = X.mean(axis=0)
    icov = np.linalg.inv(cov)
    T2 = N * ((Xbar - means) @ icov @ (Xbar - means))
    p = stats.f.pdf(T2, dfn=(dim * (N-1)), dfd=(N - dim))
    return T2, p


def hotelling_T2(sample, reference, cov=None):
    """Hotelling's T**2 test for multivariate equal means"""
    if reference.ndim == 2 and reference.shape[1] == sample.shape[1]:
        means = reference.mean(axis=0)
        cov = np.cov(reference.T)
    else:
        means = reference
    return _simple_T2(sample, means, cov)
