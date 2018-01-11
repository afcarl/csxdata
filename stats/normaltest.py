from ..utilities import sanity


def _translate(p, alpha):
    return "not normal" if p < alpha else "normal"


def _printfinds(ps, testname, names, alpha):
    if names is None:
        names = [str(i) for i in range(1, len(ps)+1)]
    print("-"*50)
    print(f"{testname} univariates:")
    for p, n in zip(ps, names):
        print(f"Feature {n} is {_translate(p, alpha)} (p = {p:.4f})")


def skewkurt(data):
    """From skewness and curtosis information"""
    from scipy.stats import normaltest
    return normaltest(data, axis=0).pvalue


def ks(data):
    """Kolmogorov-Smirnov test of normality"""
    from scipy.stats import kstest

    nfeatures = data.shape[1]
    return [kstest(data[:, i], "norm").pvalue for i in range(nfeatures)]


def sw(data):
    """Shapiro-Wilk test of normality"""
    from scipy.stats import shapiro
    nfeatures = data.shape[1]
    return [shapiro(data[:, i])[1] for i in range(nfeatures)]


def full(data, alpha=0.05, names=None):
    """Runs all tests of normality"""
    X, nm = sanity.asmatrix(data, getnames=True, matrixwarn=True)
    names = nm if names is None else names
    skps = skewkurt(X)
    ksps = ks(X)
    swps = sw(X)

    _printfinds(skps, "Skewness-Kurtosis", names, alpha)
    _printfinds(ksps, "Kolmogorov-Smirnov", names, alpha)
    _printfinds(swps, "Shapiro-Wilk's", names, alpha)
