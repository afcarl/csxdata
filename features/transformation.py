import numpy as np

from ..utilities.vectorop import dummycode
from ..utilities.highlevel import transform


class _Transformation:

    name = ""

    def __init__(self, factors=None):
        self.factors = factors
        self.model = None
        self._transformation = None
        self._transform = None
        self._applied = False

    def fit(self, X, Y=None):
        self.model = transform(X, self.factors, get_model=True, method=self.name, y=Y)[-1]

    def _apply(self, X, Y=None):
        return self.model.transform(X)[..., :self.factors]

    def __str__(self):
        return self.name

    def __call__(self, X, Y=None):
        return self._apply(X, Y)


class Standardization(_Transformation):

    name = "std"

    def fit(self, X, Y=None):
        from ..utilities.vectorop import standardize
        self.model = standardize(X, return_factors=True)[1]

    def _apply(self, X: np.ndarray, Y=None):
        mean, std = self.model
        return (X - mean) / std


class PCA(_Transformation):
    name = "pca"


class LDA(_Transformation):
    name = "lda"


class ICA(_Transformation):
    name = "ica"


class PLS(_Transformation):
    name = "pls"

    def _apply(self, X: np.ndarray, Y=None):
        Y = dummycode(Y, get_translator=False)
        ret = self.model.transform(X, Y)[0]
        return ret


class Autoencoding(_Transformation):
    """
    Performs Autoencoding on the data for dimension transformation.
    Currently wraps Keras, but I intend to switch to CsxNet backend.
    """
    def __init__(self, features, epochs=5):
        self.epochs = epochs
        _Transformation.__init__(features)

    def fit(self, X, Y=None):
        from ..utilities.highlevel import autoencode
        self.model = autoencode(X, self.factors, epochs=self.epochs, get_model=True)[1:]

    def _apply(self, X: np.ndarray, Y=None):
        (encoder, decoder), (mean, std) = self.model[0], self.model[1]
        X = np.copy(X)
        X -= mean
        X /= std
        for weights, biases in encoder:
            X = np.tanh(X.dot(weights) + biases)
        return X
