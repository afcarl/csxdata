import numpy as np

from const import *


def parse_csv(path: str, header: int, indeps_n: int, sep: str, end: str):
    """Extracts a data table from a file

    Returns data, header indeps_n"""

    header, indeps_n = int(header), int(indeps_n)

    def load_from_file_to_array():
        with open(path) as f:
            text = f.read()
            f.close()
        assert sep in text and end in text, "Separator or Endline character not present in file!"
        if "," in text:
            print("Warning! Replacing every ',' character with '.'!")
            text = text.replace(",", ".")
        return np.array([l.split(sep) for l in text.split(end) if l])

    lines = load_from_file_to_array()
    lines, header, indeps = parse_array(lines, header, indeps_n)
    return lines, header, indeps


def parse_array(X: np.ndarray, header: int, indeps_n: int):
    header, indeps_n = int(header), int(indeps_n)
    headers = X[:header] if header else None
    matrix = X[header:] if header else X
    indeps = matrix[:, :indeps_n]
    data = matrix[:, indeps_n:].astype(floatX)
    return data, headers, indeps


def parse_learningtable(source, coding=None):
    if isinstance(source, str) and source[-7:] == ".pkl.gz":
        source = load_learningtable(source, coding)
    if not isinstance(source, tuple):
        raise RuntimeError("Please supply a learning table (tuple) or a *lt.pkl.gz file!")
    if source[0].dtype != floatX:
        source = source[0].astype(floatX), source[1]
        print("Warning! dtype differences in datamodel.parselearningtable()!\n" +
              "Casting <{}> to <{}>".format(source[0].dtype, floatX))

    return None, source[0], source[1]


def load_learningtable(source: str, coding='latin1'):
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

# TODO: write tests!
