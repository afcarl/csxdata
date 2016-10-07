"""
Some features as classes for data manipulation.
Transformations are used to transform the independent variables (X)
Embeddings are used to transform the dependent variables or categories (y)
"""

import abc
import warnings

import numpy as np


###################
# Transformations #
###################


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
        er = "Please supply the number of factors (> 0) as <param> for PCA/LDA!"
        if self.name[0] == "s":
            if self.param:
                warnings.warn("Supplied parameters but chose standardization! Parameters are ignored!",
                              RuntimeWarning)
        elif self.name[0] == "p" or "l" or "i":
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
        _Transformation.__init__(self, "standardization", None)

    def fit(self, X, y=None):
        from .nputils import standardize
        self._model = standardize(X, return_factors=True)[1]

    def _apply(self, X: np.ndarray):
        mean, std = self._model
        return (X - mean) / std


class PCA(_Transformation):
    def __init__(self, factors):
        _Transformation.__init__(self, "pca", params=factors)

    def fit(self, X, y=None):
        from .high_utils import pca_transform

        self._model = pca_transform(X, self.param, get_model=True)[-1]

    def _apply(self, X):
        return self._model.transform(X)[..., :self.param]


class LDA(_Transformation):
    def __init__(self, factors):
        _Transformation.__init__(self, "lda", params=factors)

    def fit(self, X, y):
        from .high_utils import lda_transform
        self._model = lda_transform(X, y, factors=self.param, get_model=True)[-1]

    def _apply(self, X: np.ndarray):
        return self._model.transform(X)[..., :self.param]


class ICA(_Transformation):
    def __init__(self, factors):
        _Transformation.__init__(self, "ica", params=factors)

    def fit(self, X, y=None):
        from .high_utils import ica_transform
        self._model = ica_transform(X, factors=self.param, get_model=True)[-1]

    def _apply(self, X: np.ndarray):
        return self._model.transform(X)[..., :self.param]


class Autoencoding(_Transformation):
    def __init__(self, features, epochs=5):
        self.epochs = epochs
        _Transformation.__init__(self, "autoencoding", features)

    def fit(self, X, y=None):
        from .high_utils import autoencode

        self._model = autoencode(X, self.param, epochs=self.epochs, get_model=True)[1:]

    def _apply(self, X: np.ndarray):
        (encoder, decoder), (mean, std) = self._model[0], self._model[1]
        X = np.copy(X)
        X -= mean
        X /= std
        for weights, biases in encoder:
            X = np.tanh(X.dot(weights) + biases)
        return X


class Transformation:
    @staticmethod
    def pca(factors):
        return PCA(factors)

    @staticmethod
    def autoencoder(features):
        return Autoencoding(features)

    @staticmethod
    def standardization(features=None):
        del features
        return Standardization()

    @staticmethod
    def lda(features=None):
        return LDA(features)

    @staticmethod
    def ica(features=None):
        return ICA(factors=features)


##############
# Embeddings #
##############


class _Embedding(abc.ABC):
    def __init__(self, name):
        self.name = name
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
    def __init__(self, yes=None, no=None):
        _Embedding.__init__(self, name="onehot")

        from .const import YAY, NAY
        self._yes = YAY if yes is None else yes
        self._no = NAY if no is None else no

        self.dim = 0

    def translate(self, prediction: np.ndarray, dummy: bool=False):
        if prediction.ndim == 2:
            prediction = np.argmax(prediction, axis=1)
            if dummy:
                return prediction

        return _Embedding.translate(self, prediction)

    def fit(self, X):
        _Embedding.fit(self, X)

        self.dim = len(self._categories)

        self._embedments = np.zeros((self.dim, self.dim)) + self._no
        np.fill_diagonal(self._embedments, self._yes)

        self.outputs_required = self.dim
        self._fitted = True


class Embed(_Embedding):
    def __init__(self, embeddim):
        _Embedding.__init__(self, name="embedding")

        self.dim = embeddim

    def translate(self, prediction: np.ndarray, dummy: bool=False):
        from .nputils import euclidean
        if prediction.ndim > 2:
            raise RuntimeError("<prediction> must be a matrix!")

        dummycodes = [np.argmin(euclidean(pred, self._embedments)) for pred in prediction]
        if dummy:
            return dummycodes

        return _Embedding.translate(self, dummycodes)

    def fit(self, X):
        _Embedding.fit(self, X)
        cats = len(self._categories)

        self._embedments = np.random.randn(cats, self.dim)
        self.outputs_required = self.dim
        self._fitted = True


class Embedding:
    @classmethod
    def onehot(cls, yes=None, no=None):
        return OneHot(yes, no)

    @classmethod
    def embed(cls, embeddim):
        return Embed(embeddim)
