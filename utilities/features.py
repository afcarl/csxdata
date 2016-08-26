"""
Some features as classes for data manipulation.
Transformations are used to transform the independent variables (X)
Embeddings are used to transform the dependent variables or categories (y)
"""


import warnings
import abc

import numpy as np


class _Transformation(abc.ABC):
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

    def _sanity_check(self):
        if self.name[0] == "s":
            if self.params:
                warnings.warn("Supplied parameters but chose standardization! Parameters are ignored!",
                              RuntimeWarning)
        elif self.name[0] == "p":
            er = "Please supply the number of factors (> 0) as <params> for PCA!"
            if not self.params:
                raise RuntimeError(er)
            if isinstance(self.params, str):
                if self.params != "full":
                    raise RuntimeError(er)
            if isinstance(self.params, int):
                if self.params <= 0:
                    raise RuntimeError(er)
            else:
                raise RuntimeError(er)

        else:
            if not self.params or not isinstance(self.params, int):
                raise RuntimeError(er)

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


class Standardization(_Transformation):
    def __init__(self, master, features=None):
        if features is not None:
            warnings.warn("Received <feautres> paramter ({}). Ignored!".format(features),
                          RuntimeWarning)
        _Transformation.__init__(self, master, "standardization", None)

    def _fit(self):
        from .nputils import standardize
        self._model = standardize(self._master.learning, return_factors=True)[1]

    def _apply(self, X: np.ndarray):
        mean, std = self._model
        return (X - mean) / std


class PCA(_Transformation):
    def __init__(self, master, factors):
        _Transformation.__init__(self, master, "pca", params=factors)

    def _fit(self):
        from .high_utils import pca_transform

        self._model = pca_transform(self._master.learning, self.params,
                                    whiten=True, get_model=True)[-1]

    def _apply(self, X):
        return self._model.transform(X)[..., :self.params]


class Autoencoding(_Transformation):
    def __init__(self, master, features, epochs=5):
        self.epochs = epochs
        _Transformation.__init__(self, master, "autoencoding", features)

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


class Transformation:
    @classmethod
    def pca(cls, master, factors):
        return PCA(master, factors)

    @classmethod
    def autoencoder(cls, master, features):
        return Autoencoding(master, features)

    @classmethod
    def standardization(cls, master, features=None):
        return Standardization(master)


class _Embedding(abc.ABC):
    def __init__(self, master, name):
        self.name = name
        self.master = master
        self._categories = None
        self._embedments = None
        self._translate = None
        self.outputs_required = None
        self.dummycode = None
        self._fitted = False

    @abc.abstractmethod
    def translate(self, X):
        return self._translate(X)

    @abc.abstractmethod
    def fit(self, X):
        self._categories = sorted(list(set(X)))
        self.dummycode = np.vectorize(lambda x: self._categories.index(x))
        self._translate = np.vectorize(lambda x: self._categories[x])

    def _apply(self, X):
        dcs = self.dummycode(X)
        return self._embedments[dcs]

    def __str__(self):
        return self.name

    def __call__(self, X):
        if not self._fitted:
            raise RuntimeError("Not yet fitted! Call fit() first!")
        return self._apply(X)


class OneHot(_Embedding):
    def __init__(self, master, yes=None, no=None):
        _Embedding.__init__(self, master, name="onehot")

        from ..const import YAY, NAY
        self._yes = YAY if yes is None else yes
        self._no = NAY if no is None else no

        self.dim = 0

    def translate(self, prediction: np.ndarray, dummy: bool=False):
        if prediction.ndim == 2:
            prediction = np.argmax(prediction, axis=0)
            if dummy:
                return prediction

        return _Embedding.translate(self, prediction)

    def fit(self, X):
        _Embedding.fit(self, X)

        cats = len(self._categories)

        self._embedments = np.zeros((cats, cats))
        self._embedments += self._no

        np.fill_diagonal(self._embedments, self._yes)

        self.outputs_required = cats
        self._fitted = True


class Embed(_Embedding):
    def __init__(self, master, embeddim):
        _Embedding.__init__(self, master, name="embedding")

        self.dim = embeddim
        self._targets = None

    def translate(self, prediction: np.ndarray, dummy: bool=False):
        from .nputils import euclidean
        if prediction.ndim != 2:
            raise RuntimeError("<prediction> must be a matrix!")

        prediction = np.argmin(euclidean(prediction, self._targets))
        if dummy:
            return prediction

        return _Embedding.translate(self, prediction)

    def fit(self, X):
        _Embedding.fit(self, X)
        cats = len(self._categories)

        self._embedments = np.random.randn(cats, self.dim)
        self.outputs_required = self.dim
        self._fitted = True


class Embedding:
    @classmethod
    def onehot(cls, master, yes=None, no=None):
        return OneHot(master, yes, no)

    @classmethod
    def embed(cls, master, embeddim):
        return Embed(master, embeddim)
