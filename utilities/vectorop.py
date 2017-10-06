"""Utility functions that use the NumPy library"""
import warnings

import numpy as np

floatX = "float32"


def rescale(X: np.ndarray, axis=0, ufctr=(0, 1), dfctr=None, return_factors=False):
    if X.ndim != 2:
        raise RuntimeError("Can only feature scale matrices!")
    if dfctr is None:
        dfctr = (X.min(axis=axis), X.max(axis=axis))
    output = upscale(downscale(X, *dfctr), *ufctr)
    return (output, dfctr, ufctr) if return_factors else output


def upscale(A, mini, maxi):
    return A * (maxi - mini) + mini


def downscale(A, mini, maxi):
    return (A - mini) / (maxi - mini)


def standardize(X, mean=None, std=None, return_factors=False):
    mean = X.mean(axis=0) if mean is None else mean
    std = (X.std(axis=0) + 1e-8) if std is None else std
    scaled = (X - mean) / std
    return (scaled, (mean, std)) if return_factors else scaled


def euclidean(itr: np.ndarray, target: np.ndarray):
    # return np.linalg.norm(itr - target, axis=0)  slower !!!
    return np.sqrt(np.sum(np.square(itr - target), axis=0))


def haversine(coords1: np.ndarray, coords2: np.ndarray):
    """Distance of two points on the surface of Earth given their GPS (WGS) coordinates"""
    err = "Please supply two arrays of coordinate-pairs!"
    assert coords1.ndim == coords2.ndim == 2, err
    assert coords1.shape == coords2.shape, err

    R = 6367.  # Approximate radius of Mother Earth in kms
    np.radians(coords1, out=coords1)
    np.radians(coords2, out=coords2)
    lon1, lat1 = coords1[..., 0], coords1[..., 1]
    lon2, lat2 = coords2[..., 0], coords2[..., 1]
    dlon = lon1 - lon2
    dlat = lat1 - lat2
    d = np.sin(dlat / 2.) ** 2. + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.) ** 2.
    e = 2. * np.arcsin(np.sqrt(d))
    return e * R


def ravel_to_matrix(A):
    A = np.atleast_2d(A)
    return A.reshape(A.shape[0], np.prod(A.shape[1:]))


def argshuffle(array):
    indices = np.arange(len(array))
    np.random.shuffle(indices)
    return indices


def shuffle(*arrays):
    indices = argshuffle(arrays[0])
    return tuple(map(lambda ar: ar[indices], arrays))


def sumsort(A: np.ndarray, axis=0):
    arg = argsumsort(A, axis=axis)
    return A[arg] if axis else A[:, arg]


def argsumsort(A: np.ndarray, axis=0):
    if A.ndim < 2:
        raise ValueError("At leas 2D array required for sumsort!")
    sums = A.sum(axis=axis)
    indices = np.argsort(sums)
    return indices


def dummycode(dependent, get_translator=True):
    categ = np.unique(dependent)
    dummy = np.arange(len(categ))

    dummy_dict = dict()
    dreverse = dict()

    applier = np.vectorize(lambda x: dummy_dict[x])
    reverter = np.vectorize(lambda x: dreverse[x])

    for c, d in zip(categ, dummy):
        dummy_dict[d] = c
        dummy_dict[c] = d

    dependent = applier(dependent)
    return (dependent, applier, reverter) if get_translator else dependent


def split_by_categories(labels: np.ndarray, X: np.ndarray=None):
    categ = np.unique(labels)
    argsbycat = {cat: np.argwhere(labels == cat).ravel() for cat in categ}
    return argsbycat if X is None else {cat: X[argsbycat[cat]] for cat in categ}


def drop_lowNs(treshold, Y, *arrays):
    categ = np.unique(Y)
    repres = np.array([(Y == cat).sum() for cat in categ])
    invalid = set(categ[repres < treshold])
    validargs = np.array([i for i, y in enumerate(Y) if y not in invalid])
    return [Y[validargs]] + [ar[validargs] for ar in arrays]


def dropna(X, *arrays):
    valid = np.unique(np.argwhere(~np.isnan(X))[:, 0])
    return [X[valid]] + [ar[valid] for ar in arrays]


def to_ngrams(txt, ngram):
    txar = np.array(list(txt))
    N = txar.shape[0]
    if N % ngram != 0:
        warnings.warn(
            "Text length not divisible by ngram. Disposed some elements at the end of the seq!",
            RuntimeWarning)
        txar = txar[:-(N % ngram)]
    txar = txar.reshape(N // ngram, ngram)
    return ["".join(ng) for ng in txar] if ngram > 1 else np.ravel(txar)


def to_wordarray(txt):
    return np.array(txt.split(" "))
