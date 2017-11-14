import warnings

import numpy as np

from csxdata import CData
from ..utilities.vectorop import ravel_to_matrix as rtm


def _prepare_data(X):
    """
    Extracts learning data from dataframe if needed and
    ravels the data to a matrix in case it's multidimensional
    """
    if not isinstance(X, np.ndarray):
        if isinstance(X, CData):
            X = X.learning
        else:
            X = X.as_matrix()
    if len(X.shape[1:]) > 1:
        warnings.warn("Normality testing on multidimensional data!", RuntimeWarning)
        X = rtm(X)
    return X


def _translate(pval, alpha):
    return ("" if pval > alpha else "not ") + "normal."


def _printfinds(ps, testname, names, alpha):
    if names is None:
        names = [str(i) for i in range(1, len(ps)+1)]
    print("-"*50)
    print(f"{testname} univariates:")
    for p, n in zip(ps, names):
        print("Feature {} is {}".format(n, _translate(p, alpha)))


def skewkurt(data):
    """From skewness and curtosis information"""
    from scipy.stats import normaltest
    return normaltest(_prepare_data(data), axis=0).pvalue


def ks(data):
    """Kolmogorov-Smirnov test of normality"""
    from scipy.stats import kstest

    X = _prepare_data(data)
    nfeatures = X.shape[1]
    return [kstest(X[:, i], "norm").pvalue for i in range(nfeatures)]


def sw(data):
    """Shapiro-Wilk test of normality"""
    from scipy.stats import shapiro
    X = _prepare_data(data)
    nfeatures = X.shape[1]
    return [shapiro(X[:, i])[1] for i in range(nfeatures)]


def full(data, alpha=0.05, names=None):
    """Runs all tests of normality"""
    skps = skewkurt(data)
    ksps = ks(data)
    swps = sw(data)

    _printfinds(skps, "Skewness-Kurtosis", names, alpha)
    _printfinds(ksps, "Kolmogorov-Smirnov", names, alpha)
    _printfinds(swps, "Shapiro-Wilk's", names, alpha)
