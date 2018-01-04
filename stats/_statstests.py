import itertools
import numpy as np
import pandas as pd

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


def pairwise_T2(X, Y, dumproot=None, xpid="", verbose=True):
    categ = np.sort(np.unique(Y))
    output = pd.DataFrame(index=categ, columns=categ)
    for a, b in itertools.combinations(np.unique(Y), 2):
        F, p = hotelling_T2(X[Y == a], X[Y == b])
        output.loc[a, b] = F
        output.loc[b, a] = p
    if verbose:
        print("-"*50)
        print(output.to_string(float_format=lambda s: f"{s:.4f}", na_rep=""))
    if dumproot is not None:
        output.to_excel(f"{dumproot}{xpid}_T2Posthoc.xlsx")
    return output
