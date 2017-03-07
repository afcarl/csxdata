import warnings

import numpy as np
from .const import floatX


class Parse:
    @staticmethod
    def csv(path, indeps=1, headers=1, **kw):
        return parse_csv(path, indeps, headers, **kw)

    @staticmethod
    def txt(source, ngram, **kw):
        return parse_text(source, n_gram=ngram, **kw)

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


def parse_csv(path: str, indeps: int=1, headers: int=1, **kw):
    """Extracts a data table from a file, returns X, Y, header"""

    get = kw.get

    def feature_name_to_index(featurename):
        if isinstance(featurename, int):
            if indeps < featurename:
                raise ValueError("Invalid feature number. Max:", indeps)
            return featurename
        elif not featurename:
            return 0

        if get("lower"):
            featurename = featurename.lower()
        try:
            got = header.tolist().index(featurename)
        except ValueError:
            raise ValueError("Unknown feature: {}\nAvailable: {}".format(featurename, header))
        return got

    def filter_data(*data):
        from .vectorops import argfilter

        if get("selection") is None:
            raise ValueError("Please supply a selection argument for filtering!")
        filterindex = feature_name_to_index(get("filterby"))
        filterargs = argfilter(data[1][:, filterindex], get("selection")).ravel()
        return data[0][filterargs], data[1][filterargs]

    def select_classification_feature(feature_matrix):
        nofeature = feature_name_to_index(get("feature", ""))
        return feature_matrix[:, nofeature]

    def load_from_file_to_array():
        with open(path, encoding=get("coding", "utf8")) as f:
            text = f.read()
        assert get("sep", "\t") in text and get("end", "\n") in text, \
            "Separator or Endline character not present in file!"
        if get("decimal"):
            text = text.replace(",", ".")
        if get("lower"):
            text = text.lower()
        return np.array([l.split(get("sep", "\t")) for l in text.split(get("end", "\n")) if l])

    headers, indeps = int(headers), int(indeps)

    lines = load_from_file_to_array()
    X, Y, header = parse_array(lines, indeps, headers, dtype=get("dtype", floatX))
    if get("shuffle"):
        from .vectorops import shuffle
        X, Y = shuffle((X, Y))
    if get("absval"):
        X = np.abs(X)
    if get("filterby") is not None:
        X, Y = filter_data(X, Y)

    Y = select_classification_feature(Y)

    if get("frame"):
        from ..frames import CData
        output = CData((X, Y), header=None)
        if headers:
            ft = get("feature", "")
            output.header = [ft.lower() if get("lower") else ft] + header[indeps:].tolist()
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
