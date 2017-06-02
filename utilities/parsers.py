import warnings

import numpy as np
from .misc import isnumber, dehungarize


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
        from .vectorops import argfilter
        filterno = self._feature_name_to_index(featurename)
        selection = np.array(selections)
        filterargs = argfilter(self.Y[:, filterno], selection)
        self.X, self.Y = self.X[filterargs], self.Y[filterargs]

    def revert(self):
        self.X, self.Y = self.raw


class Parse:
    @staticmethod
    def csv(path, indeps=1, headers=1, **kw):
        return parse_csv(path, indeps, headers, **kw)

    @staticmethod
    def txt(source, ngram, **kw):
        return parse_text(source, n_gram=ngram, **kw)

    @staticmethod
    def massive_txt(source, bsize, ngrams=1, **kw):
        return parse_text2(source, bsize, ngrams, **kw)

    @staticmethod
    def array(A, indeps=1, headers=1, dtype="float32"):
        return parse_array(A, indeps, headers, dtype)

    @staticmethod
    def learning_table(source, coding=None, dtype="float32"):
        return parse_learningtable(source, coding, dtype)


def parse_csv(path: str, indeps: int=1, headers: int=1, **kw):
    """Extracts a data table from a file, returns X, Y, header"""

    gkw = kw.get
    sep, end = gkw("sep", "\t"), gkw("end", "\n")

    headers, indeps = int(headers), int(indeps)

    with open(path, encoding=gkw("coding", "utf8")) as f:
        text = f.read()
    assert sep in text and end in text, \
        "Separator or Endline character not present in file!"
    if gkw("dehungarize"):
        text = dehungarize(text)
    if gkw("decimal"):
        text = text.replace(",", ".")
    if gkw("lower"):
        text = text.lower()
    lines = np.array([l.split(gkw("sep", "\t")) for l in text.split(gkw("end", "\n")) if l])
    X, Y, header = parse_array(lines, indeps, headers,
                               dtype=gkw("dtype", "float32"),
                               shuffle=gkw("shuffle", False))
    return reparse_data(X, Y, header, **kw)


def parse_array(A: np.ndarray, indeps: int=1, headers: int=1, dtype="float32", shuffle=False):
    headers, indeps = int(headers), int(indeps)
    header = A[:headers].ravel() if headers else None
    matrix = A[headers:] if headers else A
    Y = matrix[:, :indeps]
    X = matrix[:, indeps:]  # type: np.ndarray
    X[np.logical_not(np.vectorize(isnumber)(X))] = "nan"
    X = X.astype(dtype)
    if shuffle:
        from .vectorops import shuffle
        X, Y = shuffle((X, Y))
    return X, Y, header


def parse_learningtable(source, coding=None, dtype="float32"):
    if isinstance(source, str) and source[-7:] == ".pkl.gz":
        source = load_learningtable(source, coding)
    if not isinstance(source, tuple):
        raise RuntimeError("Please supply a learning table (tuple) or a *lt.pkl.gz file!")
    if source[0].dtype != dtype:
        # warnings.warn("dtype differences in datamodel.parselearningtable()!\n" +
        #               "Casting <{}> to <{}>".format(source[0].dtype, dtype),
        #               RuntimeWarning)
        source = source[0].astype(dtype), source[1]
    X, y = source
    return X, y, None


def parse_text(source, n_gram=1, **reparse_kw):
    from .misc import pull_text

    def reparse_as_ndarray(tx):
        tx = tx.replace("\n", " ")
        return np.array(list(tx))

    def chop_up_to_ngrams(tx, ngr):
        txar = reparse_as_ndarray(tx)
        N = txar.shape[0]
        if N % ngr != 0:
            warnings.warn("Text length not divisible by ngram. Disposed some elements at the end of the seq!",
                          RuntimeWarning)
            txar = txar[:-(N % ngr)]
        txar = txar.reshape(N // ngr, ngr)
        if ngr > 1:
            txar = ["".join(ngram) for ngram in txar]
        else:
            txar = np.ravel(txar)
        return txar

    def chop_up_to_words(tx):
        wordar = np.array(tx.split(" "))
        return wordar

    if ("\\" in source or "/" in source) and len(source) < 200:
        source = pull_text(source, **reparse_kw)
    if n_gram:
        source = chop_up_to_ngrams(source, n_gram)
    else:
        source = chop_up_to_words(source)
    return source


def parse_text2(src, bsize, ngrams=1, **kw):
    """This will be a generator function"""

    kwg = kw.get

    def chop_up_to_ngrams(txar: np.ndarray, ngr):
        N = txar.shape[0]
        if N % ngr != 0:
            raise RuntimeError("bsize ({0}) not divisible by ngrams ({1})! ({0} % {1} = {2}"
                               .format(N, ngr, N % ngr))
        txar = txar.reshape(N // ngr, ngr)
        if ngr > 1:
            txar = ["".join(ngram) for ngram in txar]
        else:
            txar = np.ravel(txar)
        return txar

    if not isinstance(src, str):
        raise TypeError("Please supply a path to a text file!")

    with open(src, mode="r", encoding=kwg("coding", "utf-8-sig")) as opensource:
        chunk = opensource.read(n=bsize)
        if not chunk:
            raise StopIteration("File ended")
        if kwg("dehungarize"):
            chunk = dehungarize(chunk)
        if kwg("endline_to_space"):
            chunk = chunk.replace("\n", " ")
        if kwg("lower"):
            chunk = chunk.lower()
        chunk = chop_up_to_ngrams(chunk, ngr=ngrams)
        yield chunk


def load_learningtable(source: str, coding='utf8'):
    import pickle
    import gzip

    f = gzip.open(source)
    if coding:
        with f:
            # noinspection PyProtectedMember
            u = pickle._Unpickler(f)
            u.encoding = coding
            tup = u.load()
        f.close()
    else:
        tup = pickle.load(f)
    return tup


def mnist_tolearningtable(source: str, fold=True, dtype="float32"):
    """The reason of this method's existance is that I'm lazy as ..."""
    tup = load_learningtable(source, coding="latin1")
    questions = np.concatenate((tup[0][0], tup[1][0], tup[2][0]))
    questions = questions.astype(dtype, copy=False)
    targets = np.concatenate((tup[0][1], tup[1][1], tup[2][1]))
    if fold:
        questions = questions.reshape((questions.shape[0], 1, 28, 28))
    return questions, targets


def reparse_data(X, Y, header, **kw):
    gkw = kw.get
    indeps = Y.shape[1]
    fter = Filterer(X, Y, header.tolist())

    if gkw("absval"):
        X = np.abs(X)
    if gkw("filterby") is not None:
        fter.filter_by(gkw("filterby"), gkw("selection"))
    if gkw("feature"):
        Y = fter.select_feature(gkw("feature"))
    if gkw("discard_nans"):
        from .vectorops import discard_NaN_rows
        X, Y = discard_NaN_rows(X, Y)
    class_treshold = gkw("discard_class_treshold", 0)
    if class_treshold:
        from .vectorops import discard_lowNs
        X, Y = discard_lowNs(X, Y, class_treshold)
    if gkw("frame"):
        from ..frames import CData
        output = CData((X, Y), header=None)
        if header:
            ft = gkw("feature", "")
            output.header = [ft.lower() if gkw("lower") else ft] + header[indeps:].tolist()
        return output
    return X, Y, header
