import abc

import numpy as np


class EmbeddingBase(abc.ABC):

    def __init__(self, name, **kw):
        self.name = name
        self._categories = None
        self._embedments = None
        self._translate = None
        self.dummycode = None
        self.floatX = kw.get("floatX", "float32")
        self.fitted = False
        self.dim = None

    @abc.abstractmethod
    def translate(self, X):
        return self._translate(X)

    @abc.abstractmethod
    def fit(self, X):
        self._categories = np.sort(np.unique(X)).tolist()  # type: list
        self.dummycode = np.vectorize(lambda x: self._categories.index(x))
        self._translate = np.vectorize(lambda x: self._categories[x])

    @property
    def outputs_required(self):
        return self.dim

    def apply(self, X):
        dcs = self.dummycode(X)
        return self._embedments[dcs]

    def __str__(self):
        return self.name

    def __call__(self, X):
        if not self.fitted:
            raise RuntimeError("Not yet fitted! Call fit() first!")
        return self.apply(X)


class OneHot(EmbeddingBase):

    def __init__(self, yes=0., no=1.):
        EmbeddingBase.__init__(self, name="onehot")
        self._yes = yes
        self._no = no
        self.dim = 0

    def translate(self, prediction: np.ndarray, dummy: bool=False):
        if prediction.ndim == 2:
            prediction = np.argmax(prediction, axis=1)
            if dummy:
                return prediction

        return EmbeddingBase.translate(self, prediction)

    def fit(self, X):
        EmbeddingBase.fit(self, X)

        self.dim = len(self._categories)

        self._embedments = np.zeros((self.dim, self.dim)) + self._no
        np.fill_diagonal(self._embedments, self._yes)
        self._embedments = self._embedments.astype(self.floatX)

        self.fitted = True
        return self


class Embedding(EmbeddingBase):

    def __init__(self, embeddim):
        EmbeddingBase.__init__(self, name="embedding")

        self.dim = embeddim

    def translate(self, prediction: np.ndarray, dummy: bool=False):
        from ..utilities.vectorop import euclidean
        if prediction.ndim > 2:
            raise RuntimeError("<prediction> must be a matrix!")

        dummycodes = [np.argmin(euclidean(pred, self._embedments)) for pred in prediction]
        if dummy:
            return dummycodes

        return EmbeddingBase.translate(self, dummycodes)

    def fit(self, X):
        EmbeddingBase.fit(self, X)
        cats = len(self._categories)

        self._embedments = np.random.randn(cats, self.dim)
        self.fitted = True


def embedding_factory(embeddim, **kw):
    if not embeddim or embeddim == "onehot":
        return OneHot(**kw)
    return Embedding(embeddim)
