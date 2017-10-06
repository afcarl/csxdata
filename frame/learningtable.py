import numpy as np

from ..parser import parser


class LearningTable:

    def __init__(self, X=None, Y=None):
        self.X = X  # type: np.ndarray
        self.Y = Y  # type: np.ndarray

    @classmethod
    def parse_source(cls, source, indeps=1, headers=1, **kw):
        X, Y, header = parser.parse_source(source, indeps, headers, **kw)
        return cls(X, Y), header

    @classmethod
    def from_tuple(cls, source):
        X, Y, _ = parser.learningtable(source)
        return cls(X, Y)

    def batch_stream(self, size, infinite=True, randomize=True):
        arg = self._get_arguments_for_subsampling(randomize)
        while 1:
            if randomize:
                np.random.shuffle(arg)
            for start in range(0, len(self), size):
                yield self.X[start:start+size], self.Y[start:start+size]
            if not infinite:
                break

    def apply_transformation(self, trobj):
        if self.X is None:
            return
        self.X = trobj.transform(self.X, self.Y)

    def _get_arguments_for_subsampling(self, randomize):
        arg = np.arange(self.N)
        if randomize:
            np.random.shuffle(arg)
        return arg

    def split(self, alpha):
        if alpha == 0:
            return self.copy(), self.__class__()
        if isinstance(alpha, float):
            alpha = int(alpha * self.N)
        arg = self._get_arguments_for_subsampling(randomize=True)
        if self.X is None:
            raise RuntimeError("Empty learning table!")
        X1, X2 = self.X[arg[:alpha]], self.X[arg[alpha:]]
        Y1, Y2 = (self.Y[arg[:alpha]], self.Y[arg[alpha:]])\
            if self.Y is not None else (None, None)
        lt1, lt2 = self.__class__((X1, Y1)), self.__class__((X2, Y2))
        return lt1, lt2

    def subsample(self, m, randomize=True):
        if isinstance(m, float):
            m = int(m * self.N)
        arg = self._get_arguments_for_subsampling(randomize)
        return self.__class__(self.X[arg[:m]], self.Y[arg[:m]])

    def copy(self):
        return self.__class__(self.X, self.Y)

    def __iter__(self):
        for member in (self.X, self.Y):
            yield member

    def __getitem__(self, item):
        if isinstance(item, int):
            return (self.X, self.Y)[item]
        if isinstance(item, str):
            return self.__dict__[item]

    def __len__(self):
        return len(self.X)
