import numpy as np

from .abstract_frame import Frame
from ..utilities.vectorops import featscale


class RData(Frame):
    """
    Class for holding regression type data.
    """

    def __init__(self, source, indeps=1, headers=None, cross_val=0.2, **kw):
        Frame.__init__(self, source, cross_val, indeps, headers, **kw)

        self.type = "regression"
        self._downscaled = False

        self._oldfctrs = None
        self._newfctrs = None

        self.indeps = np.atleast_2d(self.indeps)

        transformation = kw.get("transformation")
        trparameter = kw.get("trparam")
        self.set_transformation(transformation, trparameter)

        self.reset_data(shuff=False, transform=False, trparam=None)

    def reset_data(self, shuff=True, transform=None, trparam=None):
        Frame.reset_data(self, shuff, transform, trparam)
        if not self._downscaled:
            self.lindeps, self._oldfctrs, self._newfctrs = \
                featscale(self.lindeps, axis=0, ufctr=(0.1, 0.9), return_factors=True)
            self._downscaled = True
            if self.n_testing:
                self.tindeps = self.downscale(self.tindeps)
                self.tindeps = self.tindeps.astype(self.floatX)

        self.indeps = self.indeps.astype(self.floatX)
        self.lindeps = self.lindeps.astype(self.floatX)

    def _scale(self, A, where):

        def sanitize():
            if not self._downscaled:
                raise RuntimeError("Scaling factors not yet set!")
            if where == "up":
                return self._newfctrs, self._oldfctrs
            else:
                return self._oldfctrs, self._newfctrs

        downfactor, upfactor = sanitize()
        return featscale(A, axis=0, dfctr=downfactor, ufctr=upfactor)

    def upscale(self, A):
        return self._scale(A, "up")

    def downscale(self, A):
        return self._scale(A, "down")

    @property
    def neurons_required(self):
        fanin, outshape = self.learning.shape[1:], self.lindeps.shape[1]
        if len(fanin) == 1:
            fanin = fanin[0]
        return fanin, outshape

    def concatenate(self, other):
        transform, trparam = Frame.concatenate(self, other)
        self.data = np.concatenate((self.data, other.data))
        self.indeps = np.concatenate((self.indeps, other.indeps))
        self._downscaled = False
        self.reset_data(shuff=False, transform=transform, trparam=trparam)