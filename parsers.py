

def parsecsv(source: str, header: int, indeps_n: int, sep: str, end: str):
    file = open(source, encoding='utf8')
    text = file.read()
    text = text.replace(",", ".")
    file.close()
    assert sep in text and end in text, \
        "Separator or Endline character is not present in the file!"

    lines = text.split(end)

    if header:
        headers = lines[0]
        headers = headers.split(sep)
        lines = lines[1:-1]
    else:
        lines = lines[:-1]
        headers = None

    lines = np.array([line.split(sep) for line in lines])
    indeps = lines[..., :indeps_n]
    data = lines[..., indeps_n:].astype(floatX)

    return headers, data, indeps


def parsearray(X: np.ndarray, header: int, indeps_n: int):
    headers = X[0] if header else None
    matrix = X[1:] if header else X
    indeps = matrix[:, :indeps_n]
    data = matrix[:, indeps_n:]
    return headers, data, indeps


def parselearningtable(source, coding=None):
    if isinstance(source, str) and source[-7:] == ".pkl.gz":
        source = unpickle_gzip(source, coding)
    if not isinstance(source, tuple):
        raise RuntimeError("Please supply a learning table (tuple) or a *lt.pkl.gz file!")
    if source[0].dtype != floatX:
        print("Warning! dtype differences in datamodel.parselearningtable()! Casting...")
        source = source[0].astype(floatX), source[1]

    return None, source[0], source[1]


def unpickle_gzip(source: str, coding='latin1'):
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


def mnist_to_lt(source: str, reshape=True):
    """The reason of this method's existance is that I'm lazy as ..."""
    tup = unpickle_gzip(source, coding="latin1")
    questions = np.concatenate((tup[0][0], tup[1][0], tup[2][0]))
    questions = questions.astype(floatX, copy=False)
    targets = np.concatenate((tup[0][1], tup[1][1], tup[2][1]))
    if reshape:
        questions = questions.reshape((questions.shape[0], 1, 28, 28))
    return questions, targets
