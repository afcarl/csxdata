import warnings

import numpy as np
from .const import floatX


class Parse:
    @staticmethod
    def csv(path, indeps=1, headers=1, sep="\t", end="\n", dtype=floatX):
        return parse_csv(path, indeps, headers, sep, end, dtype)

    @staticmethod
    def txt(source, ngram, coding="utf-8-sig", dehungarize=False):
        return parse_text(source, n_gram=ngram, coding=coding, dehungarize=dehungarize)

    @staticmethod
    def massive_txt(source, bsize, ngrams=1, coding="utf-8-sig",
                    dehungarize=False, endline_to_space=False, lower=False):
        return parse_text2(source, bsize, ngrams, coding,
                           dehungarize, endline_to_space, lower)

    @staticmethod
    def array(A, indeps=1, headers=1, dtype=floatX):
        return parse_array(A, indeps, headers, dtype)

    @staticmethod
    def learning_table(source, coding=None, dtype=floatX):
        return parse_learningtable(source, coding, dtype)


def parse_csv(path: str, indeps: int=1, headers: int=1,
              sep: str="\t", end: str="\n", shuffle=False,
              dtype=floatX, lower=False, frame=False,
              feature="", filterby=None, selection=None):
    """Extracts a data table from a file, returns X, Y, header"""

    def feature_name_to_index(featurename):
        if isinstance(featurename, int):
            if indeps < featurename:
                raise ValueError("Invalid feature number. Max:", indeps)
            return featurename
        elif not featurename:
            return 0

        if lower:
            featurename = featurename.lower()
        try:
            got = header.tolist().index(featurename)
        except ValueError:
            raise ValueError("Unknown feature: {}".format(featurename))
        return got

    def filter_data(*data):
        from .vectorops import argfilter

        if selection is None:
            raise ValueError("Please supply a selection argument for filtering!")
        filterindex = feature_name_to_index(filterby)
        filterargs = argfilter(data[1][:, filterindex], selection).ravel()
        return data[0][filterargs], data[1][filterargs]

    def select_classification_feature(feature_matrix):
        nofeature = feature_name_to_index(feature)
        return feature_matrix[:, nofeature]

    def load_from_file_to_array():
        with open(path) as f:
            text = f.read()
            f.close()
        assert sep in text and end in text, "Separator or Endline character not present in file!"
        if "," in text:
            warnings.warn("Replacing every ',' character with '.'!", RuntimeWarning)
            text = text.replace(",", ".")
        if lower:
            text = text.lower()
        return np.array([l.split(sep) for l in text.split(end) if l])

    headers, indeps = int(headers), int(indeps)

    lines = load_from_file_to_array()
    X, Y, header = parse_array(lines, indeps, headers, dtype=dtype)
    if shuffle:
        from .vectorops import shuffle
        X, Y = shuffle((X, Y))

    if filterby is not None:
        X, Y = filter_data(X, Y)

    Y = select_classification_feature(Y)

    if frame:
        from ..frames import CData
        output = CData((X, Y), header=None)
        if headers:
            output.header = [feature.lower() if lower else feature] + header[indeps:].tolist()
        return output

    return X, Y, header


def parse_array(A: np.ndarray, indeps: int=1, headers: int=1,
                dtype=floatX):
    headers, indeps = int(headers), int(indeps)
    header = A[:headers].ravel() if headers else None
    matrix = A[headers:] if headers else A
    y = matrix[:, :indeps]
    X = matrix[:, indeps:].astype(dtype)
    return X, y, header


def parse_learningtable(source, coding=None, dtype=floatX):
    if isinstance(source, str) and source[-7:] == ".pkl.gz":
        source = load_learningtable(source, coding)
    if not isinstance(source, tuple):
        raise RuntimeError("Please supply a learning table (tuple) or a *lt.pkl.gz file!")
    if source[0].dtype != dtype:
        warnings.warn("dtype differences in datamodel.parselearningtable()!\n" +
                      "Casting <{}> to <{}>".format(source[0].dtype, dtype),
                      RuntimeWarning)
        source = source[0].astype(dtype), source[1]
    X, y = source
    return X, y, None


def parse_text(source, n_gram=1, coding="utf-8-sig", dehungarize=False):
    """Characterwise parsing of text"""

    def pull_text(src):
        if not isinstance(src, str):
            raise TypeError("Please supply a text source for parse_text() (duh)")
        if ("/" in src or "\\" in src) and len(src) < 200:
            with open(src, mode="r", encoding=coding) as opensource:
                src = opensource.read()
                opensource.close()
        return src.lower()

    def reparse_as_ndarray(tx, dehun):
        tx = tx.replace("\n", " ")
        if dehun:
            from .misc import dehungarize
            tx = dehungarize(tx)
        return np.array(list(tx))

    def chop_up_to_ngrams(txar: np.ndarray, ngr):
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

    source = pull_text(source)
    source = reparse_as_ndarray(source, dehun=dehungarize)
    source = chop_up_to_ngrams(source, n_gram)

    return source


def parse_text2(src, bsize, ngrams=1, coding="utf-8-sig",
                dehungarize=False, endline_to_space=False, lower=False):
    """This will be a generator function"""

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

    with open(src, mode="r", encoding=coding) as opensource:
        chunk = opensource.read(n=bsize)
        if not chunk:
            raise StopIteration("File ended")
        if dehungarize:
            from .misc import dehungarize
            chunk = dehungarize(chunk)
        if endline_to_space:
            chunk = chunk.replace("\n", " ")
        if lower:
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


def mnist_tolearningtable(source: str, fold=True):
    """The reason of this method's existance is that I'm lazy as ..."""
    tup = load_learningtable(source, coding="latin1")
    questions = np.concatenate((tup[0][0], tup[1][0], tup[2][0]))
    questions = questions.astype(floatX, copy=False)
    targets = np.concatenate((tup[0][1], tup[1][1], tup[2][1]))
    if fold:
        questions = questions.reshape((questions.shape[0], 1, 28, 28))
        print("Folded MNIST data to {}".format(questions.shape))
    return questions, targets
