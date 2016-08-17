import warnings
import abc

import numpy as np


class Transformation(abc.ABC):
    def __init__(self, master, name: str, params=None):
        self.name = name
        self.params = params
        self._master = master
        self._model = None
        self._transformation = None
        self._transform = None
        self._applied = False

        self._sanity_check()
        self._fit()

    @classmethod
    def pca(cls, master, factors):
        return PCA(master, factors)

    @classmethod
    def autoencoder(cls, master, features):
        return Autoencoding(master, features)

    @classmethod
    def standardization(cls, master):
        return Standardization(master)

    def _sanity_check(self):
        if self.name[0] == "s":
            if self.params is not None:
                warnings.warn("Supplied parameters but chose standardization! Parameters are ignored!",
                              RuntimeWarning)
        elif self.name[0] == "p":
            if any((self.params is None,
                    not (isinstance(self.params, int) or self.params == "full"),
                    0 >= self.params)):
                raise RuntimeError("Please supply the number of factors as <params> for PCA!")
        else:
            if any((self.params is None,
                    not isinstance(self.params, int),
                    0 >= self.params)):
                raise RuntimeError("Please supply the number of features as <params> for autoencoding!")

    @abc.abstractmethod
    def _fit(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _apply(self, X: np.ndarray):
        raise NotImplementedError

    def __str__(self):
        return self.name

    def __call__(self, X):
        return self._apply(X)


class Standardization(Transformation):
    def __init__(self, master, features=None):
        if features is not None:
            warnings.warn("Received <feautres> paramter ({}). Ignored!".format(features),
                          RuntimeWarning)
        Transformation.__init__(self, master, "standardization", None)

    def _fit(self):
        from .nputils import standardize
        self._model = standardize(self._master.learning, return_factors=True)[-2:]

    def _apply(self, X: np.ndarray):
        mean, std = self._model
        return (X - mean) / std


class PCA(Transformation):
    def __init__(self, master, factors):
        Transformation.__init__(self, master, "pca", params=factors)

    def _fit(self):
        from .high_utils import pca_transform

        self._model = pca_transform(self._master.learning, self.params,
                                    whiten=True, get_model=True)[-1]

    def _apply(self, X):
        return self._model.transform(X)[..., :self.params]


class Autoencoding(Transformation):
    def __init__(self, master, features, epochs=5):
        self.epochs = epochs
        Transformation.__init__(self, master, "autoencoding", features)

    def _fit(self):
        from .high_utils import autoencode

        self._model = autoencode(self._master.learning, self.params, epochs=self.epochs,
                                 validation=self._master.testing, get_model=True)[1:]

    def _apply(self, X: np.ndarray):
        (encoder, decoder), (mean, std) = self._model[0], self._model[1]
        X = np.copy(X)
        X -= mean
        X /= std
        for weights, biases in encoder:
            X = np.tanh(X.dot(weights) + biases)
        return X
