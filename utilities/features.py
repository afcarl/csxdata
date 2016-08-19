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


class Standardization(_Transformation):
    def __init__(self, master, features=None):
        if features is not None:
            warnings.warn("Received <feautres> paramter ({}). Ignored!".format(features),
                          RuntimeWarning)
        _Transformation.__init__(self, master, "standardization", None)

    def _fit(self):
        from .nputils import standardize
        self._model = standardize(self._master.learning, return_factors=True)[-2:]

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
    def standardization(cls, master):
        return Standardization(master)


class _Embedding(abc.ABC):
    def __init__(self, master, name):
        self.name = name
        self.master = master
        self._categories = None
        self._embedments = None
        self._dummycodes = None
        self.neurons_required = None

    @abc.abstractmethod
    def _fit(self):
        self._categories = list(set(self.master.indeps))
        self._dummycodes = list(range(len(self._categories)))
        self.dummycode = lambda X: self._dummycodes[]

    @abc.abstractmethod
    def translate(self, prediction: np.ndarray, dummy: bool):
        raise NotImplementedError()

    def __str__(self):
        return self.name

    def __call__(self, X):
        return self._apply(X)


class OneHot(_Embedding):
    def __init__(self, master, yes=None, no=None):
        _Embedding.__init__(self, master, name="onehot")

        from ..const import YAY, NAY
        self._yes = YAY if yes is None else yes
        self._no = NAY if no is None else no

        self._fit()

    def translate(self, prediction: np.ndarray, dummy: bool=False):
        if prediction.ndim == 2:
            np.argmax(prediction, axis=0, out=prediction)
            if dummy:
                return prediction

        if prediction.ndim != 1:
            raise RuntimeError("<preds> should be a vector containing class dummycodes!")

        return self.dummycode(prediction)

    def _fit(self):
        _Embedding._fit(self)
        cats = len(self._categories)

        self.targets = np.zeros(cats, cats)
        self.targets += self._no

        np.fill_diagonal(self.targets, self._yes)

        self._dict = dict(zip(self._categories, self.targets))
        self.neurons_required = cats


class Embed(_Embedding):
    def __init__(self, master, embeddim):
        _Embedding.__init__(self, master, name="embed")

        self._dim = embeddim
        self._fit()
        self._targets = None

    def translate(self, prediction: np.ndarray, dummy: bool=False):
        from .nputils import euclidean
        if prediction.ndim != 2:
            raise RuntimeError("<prediction> must be a matrix!")

        prediction = np.argmin(euclidean(prediction, self._targets))

    def _fit(self):
        _Embedding._fit(self)
        cats = len(self._categories)

        self._targets = np.random.randn(cats, self._dim)
        self._dict = dict(zip(self._categories, self._targets))
        self.neurons_required = self._dim


class Embedding:
    @classmethod
    def onehot(cls, master, yes=None, no=None):
        return OneHot(master, yes, no)

    @classmethod
    def embed(cls, master, embeddim):
        return Embed(master, embeddim)
