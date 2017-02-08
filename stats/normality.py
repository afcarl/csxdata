import warnings

import numpy as np

from csxdata import CData

from ..utilities.vectorops import ravel_to_matrix as rtm


def _prepare_data(X):
    """
    Extracts learning data from dataframe if needed and
    ravels the data to a matrix in case it's multidimensional
    """
    if isinstance(X, CData):
        X = X.learning
    if len(X.shape[1:]) > 1:
        warnings.warn("Normality testing on multidimensional data!", RuntimeWarning)
        X = rtm(X)
    return X


def _translate(pval, alpha):
    return ("" if pval > alpha else "not ") + "normal."


def skewkurt(data, alpha=0.05):
    """From skewness and curtosis information"""
    from scipy.stats import normaltest

    X = _prepare_data(data)
    ps = normaltest(X, axis=0).pvalue

    print("-"*50)
    print("Skewness-Kurtosis normality Univariates:")
    for i, p in enumerate(ps, start=1):
        print("Feature {} is {}".format(i, _translate(p, alpha)))

    return np.greater_equal(ps, alpha)


def ks(data, alpha=0.05):
    """Kolmogorov-Smirnov test of normality"""
    from scipy.stats import kstest

    X = _prepare_data(data)
    nfeatures = X.shape[1]
    ps = [kstest(X[:, i], "norm").pvalue for i in range(nfeatures)]

    print("-"*50)
    print("Kolmogorov-Smirnov's Univariates:")
    for i, p in enumerate(ps, start=1):
        print("Feature {} is {}".format(i, _translate(p, alpha)))

    return np.greater_equal(ps, alpha)


def sw(data, alpha=0.05):
    """Shapiro-Wilk test of normality"""
    from scipy.stats import shapiro

    X = _prepare_data(data)
    nfeatures = X.shape[1]
    ps = [shapiro(X[:, i])[1] for i in range(nfeatures)]

    print("-"*50)
    print("Shapiro-Wilk's Univariates:")
    for i, p in enumerate(ps, start=1):
        print("Feature {} is {}".format(i, _translate(p, alpha)))

    return np.greater_equal(ps, alpha)


def ad(data, alpha=0.05):
    """Anderson-Darling test of normality"""
    from scipy.stats import anderson

    try:
        critval = [0.15, 0.1, 0.05, 0.025, 0.01].index(alpha)
    except ValueError:
        raise ValueError("Acceptable alpha-values are 0.15, 0.1, 0.05, 0.025, 0.01.")

    X = _prepare_data(data)
    nfeatures = X.shape[1]
    passes = [anderson(X[:, i], "norm")[0] < critval for i in range(nfeatures)]

    print("-"*50)
    print("Anderson-Darling's Univariates:")
    for i, p in enumerate(passes, start=1):
        print("Feature {} is {}".format(i, ("" if p else "not ") + "normal"))

    return np.array(passes)


def full(data, alpha=0.05):
    """Runs all tests of normality"""
    skewkurt(data, alpha)
    ks(data, alpha)
    sw(data, alpha)
    ad(data, alpha)
