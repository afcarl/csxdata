import abc
import warnings

import numpy as np


from ..utilities.high_utils import transform


class _Transformation(abc.ABC):
    def __init__(self, name: str, params=None):
        self.name = name
        self.param = params
        self._model = None
        self._transformation = None
        self._transform = None
        self._applied = False

        self._sanity_check()

    def _sanity_check(self):
        er = "Please supply the number of factors (> 0) as <param> for {}!".format(self.name)
        if self.name == "std":
            if self.param:
                warnings.warn("Supplied parameters but chose standardization! Parameters are ignored!",
                              RuntimeWarning)
        elif self.name == "pca" or "lda" or "ica" or "pls":
            if not self.param:
                raise RuntimeError(er)
            if isinstance(self.param, str):
                if self.param != "full":
                    raise RuntimeError(er)
            if isinstance(self.param, int):
                if self.param <= 0:
                    raise RuntimeError(er)
            else:
                raise RuntimeError(er)

        else:
            if not self.param or not isinstance(self.param, int):
                raise RuntimeError(er)

    @abc.abstractmethod
    def fit(self, X, y):
        raise NotImplementedError

    @abc.abstractmethod
    def _apply(self, X: np.ndarray):
        raise NotImplementedError

    def __str__(self):
        return self.name

    def __call__(self, X):
        return self._apply(X)


class Standardization(_Transformation):
    def __init__(self, features=None):
        if features:
            warnings.warn("Received <feautres> paramter ({}). Ignored!".format(features),
                          RuntimeWarning)
        _Transformation.__init__(self, "std", None)

    def fit(self, X, y=None):
        from ..utilities.nputils import standardize
        self._model = standardize(X, return_factors=True)[1]

    def _apply(self, X: np.ndarray):
        mean, std = self._model
        return (X - mean) / std


class PCA(_Transformation):
    def __init__(self, factors):
        _Transformation.__init__(self, "pca", params=factors)

    def fit(self, X, y=None):
        self._model = transform(X, self.param, get_model=True, method="pca")[-1]

    def _apply(self, X):
        return self._model.transform(X)[..., :self.param]


class LDA(_Transformation):
    def __init__(self, factors):
        _Transformation.__init__(self, "lda", params=factors)

    def fit(self, X, y):
        self._model = transform(X, factors=self.param, get_model=True, method="lda", y=y)[-1]

    def _apply(self, X: np.ndarray):
        return self._model.transform(X)[..., :self.param]


class ICA(_Transformation):
    def __init__(self, factors):
        _Transformation.__init__(self, "ica", params=factors)

    def fit(self, X, y=None):
        self._model = transform(X, factors=self.param, get_model=True, method="ica")[-1]

    def _apply(self, X: np.ndarray):
        return self._model.transform(X)[..., :self.param]


class PLS(_Transformation):
    def __init__(self, factors):
        _Transformation.__init__(self, name="pls", params=factors)

    def fit(self, X, y):
        self._model = transform(X, factors=self.param, method="pls", get_model=True, y=y)

    def _apply(self, X: np.ndarray):
        return self._model.transform(X)[..., :self.param]


class Autoencoding(_Transformation):
    def __init__(self, features, epochs=5):
        self.epochs = epochs
        _Transformation.__init__(self, "autoencoding", features)

    def fit(self, X, y=None):
        from ..utilities.high_utils import autoencode
        self._model = autoencode(X, self.param, epochs=self.epochs, get_model=True)[1:]

    def _apply(self, X: np.ndarray):
        (encoder, decoder), (mean, std) = self._model[0], self._model[1]
        X = np.copy(X)
        X -= mean
        X /= std
        for weights, biases in encoder:
            X = np.tanh(X.dot(weights) + biases)
        return X
