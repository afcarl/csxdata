import warnings

import numpy as np


class Filterer:

    def __init__(self, X, Y, header):
        if len(header) != X.shape[1] + Y.shape[1]:
            raise RuntimeError("Invalid header for X and Y!")
        if isinstance(header, np.ndarray):
            header = header.tolist()
        self.X = X
        self.Y = Y
        self.raw = X, Y
        self.indeps = X.shape[1]
        self.header = header

    def _feature_name_to_index(self, featurename):
        if isinstance(featurename, int):
            if self.indeps < featurename:
                raise ValueError("Invalid feature number. Max: " + str(self.indeps))
            return featurename
        elif not featurename:
            return 0
        if featurename not in self.header:
            raise ValueError("Unknown feature: {}\nAvailable: {}"
                             .format(featurename, self.header))
        if self.header.count(featurename) > 1:
            warnings.warn("Unambiguity in feature selection! Using first occurence!",
                          RuntimeWarning)
        return self.header.index(featurename)

    def select_feature(self, featurename):
        featureno = self._feature_name_to_index(featurename)
        return self.Y[:, featureno]

    def filter_by(self, featurename, *selections):
        from .vectorop import argfilter
        filterno = self._feature_name_to_index(featurename)
        selection = np.array(selections)
        filterargs = argfilter(self.Y[:, filterno], selection)
        self.X, self.Y = self.X[filterargs], self.Y[filterargs]

    def revert(self):
        self.X, self.Y = self.raw


def reparse_data(X, Y, header, **kw):
    gkw = kw.get
    fter = Filterer(X, Y, header.tolist())

    if gkw("absval"):
        X = np.abs(X)
    if gkw("filterby") is not None:
        fter.filter_by(gkw("filterby"), gkw("selection"))
    if gkw("feature"):
        Y = fter.select_feature(gkw("feature"))
    if gkw("dropna"):
        from .vectorop import dropna
        X, Y = dropna(X, Y)
    class_treshold = gkw("discard_class_treshold", 0)
    if class_treshold:
        from .vectorop import drop_lowNs
        X, Y = drop_lowNs(X, Y, class_treshold)
    return X, Y, header
