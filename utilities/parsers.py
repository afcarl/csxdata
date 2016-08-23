import numpy as np

from ..const import floatX


class Parse:
    @staticmethod
    def csv(path, indeps=1, headers=1, sep="\t", end="\n", dtype=floatX):
        return parse_csv(path, indeps, headers, sep, end, dtype)

    @staticmethod
    def array(A, indeps=1, headers=1, dtype=floatX):
        return parse_array(A, indeps, headers, dtype)

    @staticmethod
    def learning_table(source, coding=None, dtype=floatX):
        return parse_learningtable(source, coding, dtype)


def parse_csv(path: str, indeps: int=1, headers: int=1,
              sep: str="\t", end: str="\n", dtype=floatX):
    """Extracts a data table from a file

    Returns data, header indeps_n"""

    headers, indeps = int(headers), int(indeps)

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
    X, y, headers = parse_array(lines, indeps, headers, dtype=dtype)
    return X, y, headers


def parse_array(A: np.ndarray, indeps: int=1, headers: int=1,
                dtype=floatX):
    headers, indeps = int(headers), int(indeps)
    header = A[:headers] if headers else None
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
        source = source[0].astype(dtype), source[1]
        print("Warning! dtype differences in datamodel.parselearningtable()!\n" +
              "Casting <{}> to <{}>".format(source[0].dtype, dtype))
    X, y = source
    return X, y, None


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
