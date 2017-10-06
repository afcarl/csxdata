import warnings

import numpy as np

from .learningtable import LearningTable
from ..features import transformation as trmodule
from ..utilities.vectorop import shuffle


class Frame:
    """
    Base class for Data Wrappers
    Can work with learning tables (X, y), plaintext files (.txt, .csv)
     and NumPy arrays.
    Also wraps whitening and other transformations (PCA, autoencoding, standardization).
    """

    def __init__(self, source, indeps, headers, cross_val=0.2, **kw):
        self._source = source
        self._parsearg = (source, indeps, headers)
        self._reparsekw = kw
        self.transformation = None
        self.floatX = kw.get("floatX", "float32")
        self.data, self.header = LearningTable.parse_source(source, indeps, headers, **kw)
        self._subset = (LearningTable(), LearningTable())
        self.crossval = cross_val

    @property
    def X(self):
        return self.data[0] if self._learning is None else self._learning[0]

    @property
    def Y(self):
        return self.data[1] if self._learning is None else self._learning[1]

    def subset(self, name, m=None, randomize=True):
        ss = self._subset[int(name == "testing")]
        if m is None:
            m = len(ss)
        return ss.subsample(m, randomize)

    def get_learning(self, m=None, randomize=True):
        return self.subset("learning", m, randomize)

    def get_testing(self, m=None, randomize=True):
        if self._testing is None:
            raise RuntimeError("No testing data available!")
        return self.subset("testing", m, randomize)

    def set_transformation(self, transformation, features=None, **kw):
        if self.transformation is not None:
            raise RuntimeError("Already transformed! Call reset_data() first!")
        if isinstance(transformation, trmodule.Transformation):
            self.transformation = transformation
        else:
            self.transformation = trmodule.transformation_factory(transformation, features, **kw)
        self.transformation.fit(self.X, self.Y)
        for ss in self._subset:
            ss.apply_transformation(self.transformation)

    def batchgen(self, bsize, subsetname="learning", infinite=False, shuff=True):
        return self.subset(subsetname).batch_stream(infinite, randomize=shuff)

    def reset_data(self, shuff, keep_transformation=True):
        nT = self.n_testing
        dat, ind = shuffle((self.X, self.Y)) if shuff else (self.X, self.Y)

        self._learning, self._testing = np.split(dat, nT) if nT else (dat, None)
        self.lindeps, self.tindeps = np.split(ind, nT) if nT else (ind, None)

        if keep_transformation:
            self.set_transformation(self.transformation)
        else:
            self.transformation = None

    @property
    def neurons_required(self):
        raise NotImplementedError

    @property
    def N(self):
        return len(self._learning)

    @property
    def crossval(self):
        return len(self._subset[1]) / len(self.data)

    @crossval.setter
    def crossval(self, alpha):
        self._subset = self.data.split(alpha)
