import abc
import warnings

import numpy as np
from ..features import transformation as trmodule
from ..utilities.const import roots
from ..utilities.vectorop import shuffle
from ..utilities import parser


class Frame(abc.ABC):
    """
    Base class for Data Wrappers
    Can work with learning tables (X, y), plaintext files (.txt, .csv)
     and NumPy arrays.
    Also wraps whitening and other transformations (PCA, autoencoding, standardization).
    """

    def __init__(self, source, cross_val, indeps, headers, **kw):

        def parse_source():
            sourceerror = TypeError("Data wrapper doesn't support supplied data source!")

            if isinstance(source, np.ndarray):
                return parser.array(source, indeps, headers, self.floatX)
            elif isinstance(source, tuple):
                return parser.learningtable(source, self.floatX)
            else:
                import pandas as pd
                if isinstance(source, pd.DataFrame):
                    return parser.dataframe(source, indeps, **kw)
            if not isinstance(source, str):
                raise sourceerror
            if ".pkl.gz" in source.lower():
                return parser.learningtable(source, self.floatX)
            elif source.lower()[-4:] in (".csv", ".txt"):
                return parser.csv(source, indeps, headers, **kw)
            elif source.lower()[-4:] in (".xls", "xlsx"):
                return parser.xlsx(source, indeps, headers, **kw)
            else:
                raise sourceerror

        self.learning = None
        self.testing = None
        self.lindeps = None
        self.tindeps = None
        self.transformation = None
        self._transformed = False
        self._crossval = 0.0
        self.n_testing = 0
        self.floatX = kw.get("floatX", "float32")

        self._tmpdata = roots["cache"] + "tmpdata.pkl"
        self._tmpindeps = roots["cache"] + "tmpindeps.pkl"

        data, indeps, header = parse_source()
        self.data = data
        self.indeps = indeps
        self.data.flags["WRITEABLE"] = False
        if self.indeps is not None:
            self.indeps.flags["WRITEABLE"] = False

        self._determine_no_testing(cross_val)
        self._header = None if not headers else header.ravel()

    def _determine_no_testing(self, alpha):
        if alpha < 0 or alpha > len(self.data):
            raise RuntimeError("Invalid alpha!")
        if isinstance(alpha, int):
            alpha /= len(self.data)
        self._crossval = alpha
        self.n_testing = int(len(self.data) * self._crossval)

    def set_transformation(self, transformation, features):
        if self._transformed:
            self.reset_data()
        if transformation is None:
            self.reset_data(shuff=False, transform=transformation, trparam=None)
            return

        self.transformation = {
            "std": trmodule.Standardization,
            "pca": trmodule.PCA,
            "lda": trmodule.LDA,
            "ica": trmodule.ICA,
            "ae": trmodule.Autoencoding,
            "autoe": trmodule.Autoencoding,
            "pls": trmodule.PLS
        }[transformation[:5].lower()](features)

        self.transformation.fit(self.learning, self.lindeps)

        self.learning = self.transformation(self.learning, self.lindeps)
        if self.n_testing > 0:
            self.testing = self.transformation(self.testing, self.tindeps)
        self._transformed = True

    def table(self, data, m=None, shuff=False):

        data = data[0].lower()
        if data not in ("l", "t"):
            raise RuntimeError("Unkown data subset! Choose either `learning` or `testing`!")

        if data == "t" and self.n_testing == 0:
            warnings.warn("Requested testing data, which is not present!", RuntimeWarning)
            return np.array([])[None, :], np.array([])[None, :]

        X = self.learning if data == "l" else self.testing
        y = self.lindeps if data == "l" else self.tindeps

        if shuff:
            X, y = shuffle((X, y))

        return X[:m], y[:m]

    def batchgen(self, bsize, data, infinite=False, shuff=True):
        table = self.table(data, shuff=shuff)
        N = len(table)
        while 1:
            if shuff:
                shuffle(table)
            for start in range(0, N, bsize):
                yield table[0][start:start+bsize], table[1][start:start+bsize]
            if not infinite:
                break

    @abc.abstractmethod
    def reset_data(self, shuff, transform, trparam=None):

        dat, ind = shuffle((self.data, self.indeps)) if shuff else (self.data, self.indeps)

        if self.n_testing > 0:
            self.learning = dat[self.n_testing:]
            self.testing = dat[:self.n_testing]
            self.lindeps = ind[self.n_testing:] if ind is not None else None
            self.tindeps = ind[:self.n_testing] if ind is not None else None
        else:
            self.learning = dat
            self.lindeps = ind if ind is not None else None
            self.testing = None
            self.tindeps = None

        if transform is True:
            if self.transformation is None:
                return
            if trparam is None:
                self.set_transformation(self.transformation.name, self.transformation.param)
            else:
                self.set_transformation(self.transformation.name, trparam)
        elif isinstance(transform, str):
            self.set_transformation(transform, trparam)
        elif not transform or transform in ("None", None):
            self._transformed = False
            self.transformation = None
        else:
            raise RuntimeError("Specified transformation was not understood!")

    @property
    @abc.abstractmethod
    def neurons_required(self):
        raise NotImplementedError

    @property
    def dimensionality(self):
        return self.learning.shape[1:]

    @property
    def N(self):
        return len(self.learning)

    @property
    def header(self):
        if self._header is None:
            return None
        if self._header.shape[0] != np.prod(self.dimensionality) + 1:
            warnings.warn("Header does not align with dependents' shape!")
        return self._header

    @header.setter
    def header(self, head):
        if len(head) != np.prod(self.dimensionality) + 1:
            er = "Supplied header has wrong dimensionality!\n"
            er += "head: {} != {} :this".format(len(head), np.prod(self.dimensionality) + 1)
            raise RuntimeError(er)
        self._header = np.array(head)

    @property
    def paramnames(self):
        if self._header is not None:
            return self.header[1:]

    @property
    def crossval(self):
        return self._crossval

    @crossval.setter
    def crossval(self, alpha):
        self._determine_no_testing(alpha)
        self.reset_data(shuff=True, transform=True)

    @crossval.deleter
    def crossval(self):
        self._crossval = 0.0
        self.n_testing = 0

    @abc.abstractmethod
    def concatenate(self, other):
        dimerror = TypeError("Dimensions are different! Can't concatenate...")
        dtypeerror = TypeError("Data types are different! Can't concatenate...")

        if not self.data.ndim == other.data.ndim:
            raise dimerror
        if any([dim1 != dim2 for dim1, dim2 in zip(self.data.shape[1:], other.data.shape[1:])]):
            raise dimerror
        if any([dim1 != dim2 for dim1, dim2 in zip(self.indeps.shape[1:], other.indeps.shape[1:])]):
            raise dimerror
        if self.data.dtype != other.data.dtype:
            raise dtypeerror
        if self.indeps.dtype != other.indeps.dtype:
            raise dtypeerror
        if self.transformation != other.transformation:
            warnings.warn("Supplied data frames were transformed differently.")
        if self.header:
            if not all(left == right for left, right in zip(self.header, other.header)):
                warnings.warn("Frames have different headers! Header set to self.header")

        transformation = self.transformation
        if transformation:
            trparam = self.transformation.params
        else:
            trparam = None
        return transformation, trparam
