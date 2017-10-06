import numpy as np

from ..utilities.vectorop import dummycode
from ..utilities.highlevel import transform


class Transformation:

    def __init__(self, name=None, factors=None, **kw):
        self.factors = factors
        self.model = None
        self._transformation = None
        self._transform = None
        self._applied = False
        self.name = self.__class__.__name__.lower() if name is None else name

    def fit(self, X, Y=None):
        self.model = transform(X, self.factors, get_model=True, method=self.name, y=Y)[-1]

    def _apply(self, X, Y=None):
        return self.model.transform(X)[..., :self.factors]

    def __call__(self, X, Y=None):
        return self._apply(X, Y)


class Standardization(Transformation):
    name = "std"

    def fit(self, X, Y=None):
        from ..utilities.vectorop import standardize
        self.model = standardize(X, return_factors=True)[1]

    def _apply(self, X: np.ndarray, Y=None):
        mean, std = self.model
        return (X - mean) / std


class PLS(Transformation):
    name = "pls"

    def _apply(self, X: np.ndarray, Y=None):
        Y = dummycode(Y, get_translator=False)
        ret = self.model.transform(X, Y)[0]
        return ret


class Autoencoding(Transformation):
    name = "ae"

    def __init__(self, features, epochs=5):
        self.epochs = epochs
        Transformation.__init__(features)

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


def transformation_factory(name, number_of_features, **kw):
    name = name.lower()[:5]
    exc = {"std": Standardization,
           "stand": Standardization,
           "ae": Autoencoding,
           "autoe": Autoencoding,
           "pls": PLS}
    if name not in exc:
        return Transformation(name, number_of_features, **kw)
    return exc[name](number_of_features, **kw)
