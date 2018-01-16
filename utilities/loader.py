import os

import numpy as np

from .vectorop import shuffle, split_dataset, ravel_to_matrix


DEFAULT_MNIST_PATH = os.path.expanduser("~/Prog/data/mnist.pkl.gz")


def _mnist_to_learning_table(source: str=None):
    """The reason of this method's existance is that I'm lazy as ..."""
    import pickle
    import gzip

    f = gzip.open(source)
    with f:
        # noinspection PyProtectedMember
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        tup = u.load()

    questions = np.concatenate((tup[0][0], tup[1][0], tup[2][0]))
    questions = questions.astype("float32")
    targets = np.concatenate((tup[0][1], tup[1][1], tup[2][1]))
    return questions, targets


def pull_mnist_data(path=None, split=0.1, fold=False):
    if path is None:
        path = DEFAULT_MNIST_PATH

    X, Y = _mnist_to_learning_table(path)
    if not fold and X.ndim > 2:
        X = ravel_to_matrix(X)
    number_of_categories = len(np.unique(Y))
    onehot = np.eye(number_of_categories)[Y]
    if not split:
        return tuple(map(lambda ar: ar.astype("float32"), shuffle(X, onehot)))

    return tuple(map(
        lambda ar: ar.astype(float),
        split_dataset(X, onehot, ratio=split, shuff=True, normalize=True)
    ))
