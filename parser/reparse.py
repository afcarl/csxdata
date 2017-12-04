import warnings

import numpy as np
from ..utilities.misc import pull_text


class Filterer:

    def __init__(self, X, Y, header):
        if len(header) != X.shape[1] + Y.shape[1]:
            raise RuntimeError("Invalid header for X and Y!")
        if isinstance(header, np.ndarray):
            header = header.tolist()
        self.X = X
        self.Y = Y
        self.raw = X.copy(), Y.copy()
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
            warnings.warn("Ambiguity in feature selection! Using first occurence!",
                          RuntimeWarning)
        return self.header.index(featurename)

    def select_feature(self, featurename):
        featureno = self._feature_name_to_index(featurename)
        return self.Y[:, featureno]

    def filter_by(self, featurename, *selections):
        filterno = self._feature_name_to_index(featurename)
        mask = np.ones(len(self.Y), dtype=bool)
        for sel in selections:
            mask &= self.Y[:, filterno] == sel
        self.X, self.Y = self.X[mask], self.Y[mask]

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
        from ..utilities.vectorop import dropna
        X, Y = dropna(X, Y)
    class_treshold = gkw("discard_class_treshold", 0)
    if class_treshold:
        from ..utilities.vectorop import drop_lowNs
        X, Y = drop_lowNs(X, Y, class_treshold)
    return X, Y, header


def reparse_txt(src, **kw):
    replaces = kw.get("replaces", ())
    if kw.pop("decimal", False):
        replaces += ((",", "."),)
    if kw.pop("dehun", False):
        src = dehungarize(src)
    if kw.pop("lower", False):
        src = src.lower()
    for old, new in replaces:
        src = str.replace(src, old, new)
    return src


def dehungarize(src, outflpath=None, incoding=None, outcoding=None):

    hun_to_asc = {"á": "a", "é": "e", "í": "i",
                  "ó": "o", "ö": "o", "ő": "o",
                  "ú": "u", "ü": "u", "ű": "u",
                  "Á": "A", "É": "E", "Í": "I",
                  "Ó": "O", "Ö": "O", "Ő": "O",
                  "Ú": "U", "Ü": "U", "Ű": "U"}

    if ("/" in src or "\\" in src) and len(src) < 200:
        src = pull_text(src, coding=incoding)
    src = "".join(char if char not in hun_to_asc else hun_to_asc[char] for char in src)
    if outflpath is None:
        return src
    else:
        with open(outflpath, "w", encoding=outcoding) as outfl:
            outfl.write(src)
            outfl.close()
