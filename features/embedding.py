import abc

import numpy as np


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

        from ..utilities.const import YAY, NAY
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
        from ..utilities.nputils import euclidean
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
