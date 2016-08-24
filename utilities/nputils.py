"""Utility functions that use the NumPy library"""

import numpy as np

floatX = "float32"


def featscale(X: np.ndarray, axis=0, ufctr=(0, 1), dfctr=None, return_factors=False):
    """Rescales the input by first downscaling between dfctr[0] and dfctr[1], then
    upscaling it between ufctr[0] and ufctr[1]."""
    if X.ndim != 2:
        raise RuntimeError("Can only feature scale matrices!")
    if dfctr is None:
        dfctr = (X.min(axis=axis), X.max(axis=axis))
    frm, to = ufctr
    output = X - dfctr[0]
    output /= dfctr[1] - dfctr[0]
    output *= (to - frm)
    output += frm

    if not return_factors:
        return output
    else:
        return output, dfctr, ufctr


def standardize(X: np.ndarray,
                mean: np.ndarray=None, std: np.ndarray=None,
                return_factors: bool=False):
    if not ((mean is None and std is None) or (mean is not None and std is not None)):
        err = ("Please either supply the array of means AND the standard deviations for scaling,\n" +
               "or don't supply any of them. In the latter case they will be calculated.")
        raise RuntimeError(err)

    # TODO: assert shapes are OK! Maybe allow scalar mean and scalar std?
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0) + 1e-8

    scaled = (X - mean) / std

    if return_factors:
        return scaled, (mean, std)
    else:
        return scaled


def euclidean(itr: np.ndarray, target: np.ndarray):
    """Distance of points in euclidean space"""
    # return np.linalg.norm(itr - target, axis=0)  slower !!!
    return np.sqrt(np.sum(np.square(itr - target), axis=0))


def haversine(coords1: np.ndarray, coords2: np.ndarray):
    """The distance of points on the surface of Earth given their GPS (WGS) coordinates"""
    err = "Please supply two arrays of coordinate-pairs!"
    assert coords1.ndim == coords2.ndim == 2, err
    assert all([dim1 == dim2 for dim1, dim2 in zip(coords1.shape, coords2.shape)]), err

    R = 6367  # Approximate radius of Mother Earth in kms
    np.radians(coords1, out=coords1)
    np.radians(coords2, out=coords2)
    lon1, lat1 = coords1[..., 0], coords1[..., 1]
    lon2, lat2 = coords2[..., 0], coords2[..., 1]
    dlon = lon1 - lon2
    dlat = lat1 - lat2
    d = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    e = 2 * np.arcsin(np.sqrt(d))
    return e * R


def ravel_to_matrix(A):
    """Converts an ndarray to a 2d array (matrix) by keeping the first dimension as the rows
    and flattening all the other dimensions to columns"""
    if A.ndim == 2:
        return A
    A = np.atleast_2d(A)
    return A.reshape(A.shape[0], np.prod(A.shape[1:]))


def logit(Z: np.ndarray):
    """The primitive function of the sigmoid function"""
    return np.log(Z / (1 - Z))


def combination(A, W, b, scale, actfn):
    """Calculates a linear combination, then applies an activation function."""
    return actfn(A.dot(W) + b) * scale


def avg2pool(matrix):
    """Does average-pooling with stride and filter size = 2"""
    if ((matrix.shape[1] - 2) % 2) != 0:
        raise RuntimeError("Non-integer output shape!")
    osh = matrix.shape[0], (matrix.shape[1] - 2) // 2

    avg = matrix[:, ::2][:osh[1]] + \
        matrix[:, 1::2][:osh[1]]
    avg /= 2
    return avg


def avgpool(array, e, stride=None):
    """
    Pool absorbance values to reduce dimensionality.
    e := int, the size of the pooling filter
    """

    if not stride:
        stride = e
    output = np.array([])
    outsize = int(((len(array) - e) / stride) - 1)
    for n in range(outsize):
        start = n*stride
        end = start + e
        avg = np.average(array[start:end])
        output = np.append(output, avg)

    return output


def subsample(array, step):
    return array[..., ::step]


def export_to_file(path: str, data: np.ndarray, labels=None, headers=None):
    outchain = ""
    if headers is not None:
        outchain = "MA\t"
        outchain += "\t".join(headers) + "\n"
    for i in range(data.shape[0]):
        if labels is not None:
            outchain += str(labels[i]) + "\t"
        outchain += "\t".join(data[i].astype("<U11")) + "\n"
    with open(path, "w", encoding="utf8") as outfl:
        outfl.write(outchain.replace(".", ","))
        outfl.close()


def frobenius(mat, filt):
    """Calculate the Frobenius product of <filt> and <mat>.
    Meaning: compute elementwise product, then sum the resulting tensor.
    nD Array goes in, scalar comes out."""
    assert mat.shape == filt.shape, "Shapes differ! Can't multiply..."
    return np.sum(mat * filt)


def maxpool(mat):
    return np.amax(mat, axis=(0, 1))


def argshuffle(learning_table: tuple):
    shapeX, shapey = learning_table[0].shape[0], learning_table[1].shape[0]
    if shapeX != shapey:
        raise RuntimeError("Invalid learning table!")
    indices = np.arange(shapeX)
    np.random.shuffle(indices)
    return indices


def shuffle(learning_table: tuple):
    """Shuffles and recreates the learning table"""
    indices = argshuffle(learning_table)
    return learning_table[0][indices], learning_table[1][indices]


def sumsort(A: np.ndarray, axis=0):
    arg = argsumsort(A, axis=axis)
    slc = np.s_[[None] * axis + [arg]]
    return A[slc]


def argsumsort(A: np.ndarray, axis=0):
    if A.ndim < 2:
        raise ValueError("At leas 2D array required for sumsort!")
    sums = A.sum(axis=axis)
    indices = np.argsort(sums)
    return indices


def convolution(x, W, biases):
    # TODO: test this snippet!
    d = x[:, :-1, :-1].swapaxes(0, 1)
    c = x[:, :-1, 1:].swapaxes(0, 1)
    b = x[:, 1:, :-1].swapaxes(0, 1)
    a = x[:, 1:, 1:].swapaxes(0, 1)
    x = (W[:, :, 0, 0].dot(a) +
         W[:, :, 0, 1].dot(b) +
         W[:, :, 1, 0].dot(c) +
         W[:, :, 1, 1].dot(d)) + biases.reshape(-1, 1, 1)
