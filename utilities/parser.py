import numpy as np

from .reparse import reparse_data
from .misc import isnumber, dehungarize
from .vectorop import to_ngrams, to_wordarray


def array(A, indeps=1, headers=1, dtype="float32"):
    header = A[:headers].ravel() if headers else None
    matrix = A[headers:] if headers else A
    Y = matrix[:, :indeps]
    X = matrix[:, indeps:]  # type: np.ndarray
    X[~np.vectorize(isnumber)(X)] = "nan"
    X = X.astype(dtype)
    return X, Y, header


def learningtable(source, dtype="float32"):
    if isinstance(source, str) and source[-7:] == ".pkl.gz":
        import gzip
        import pickle
        source = pickle.load(gzip.open(source, "rb"))
    if not isinstance(source, tuple):
        raise RuntimeError("Please supply a tuple of (X, Y) arrays or a *.pkl.gz file!")
    X, y = source
    return X.astype(dtype), y, None


def txt(source, ngram, **kw):
    from .misc import pull_text

    if ("\\" in source or "/" in source) and len(source) < 200:
        source = pull_text(source, **kw)
    if kw.get("endline_to_space"):
        source = source.replace("\n", " ")
    if ngram:
        source = to_ngrams(np.array(list(source)), ngram)
    else:
        source = to_wordarray(source)
    return source


def massive_txt(source, bsize, ngram=1, **kw):
    kwg = kw.get

    with open(source, mode="r", encoding=kwg("coding", "utf-8-sig")) as opensource:
        chunk = opensource.read(n=bsize)
        if not chunk:
            raise StopIteration("File ended")
        if kwg("dehungarize"):
            chunk = dehungarize(chunk)
        if kwg("endline_to_space"):
            chunk = chunk.replace("\n", " ")
        if kwg("lower"):
            chunk = chunk.lower()
        chunk = to_ngrams(np.ndarray(list(chunk)), ngram)
        yield chunk


def csv(path, indeps=1, headers=1, **kw):
    """Extracts a data table from a file, returns X, Y, header"""
    gkw = kw.get
    with open(path, encoding=gkw("coding", "utf8")) as f:
        text = f.read()
    if gkw("dehungarize"):
        text = dehungarize(text)
    if gkw("decimal"):
        text = text.replace(",", ".")
    if gkw("lower"):
        text = text.lower()
    lines = np.array([l.split(gkw("sep", "\t")) for l in text.split(gkw("end", "\n")) if l])
    X, Y, header = array(lines, indeps, headers, dtype=gkw("dtype", "float32"))
    return reparse_data(X, Y, header, **kw)


def xlsx(source, indeps=1, header=1, **kw):
    import pandas as pd
    df = pd.read_excel(source, sheetname=(kw.pop("sheetname", 0)),
                       header=(header - 1) if header else None,
                       skiprows=kw.pop("skiprows", None),
                       skip_footer=kw.pop("skip_footer", 0))
    header = df.columns
    Y = df.iloc[:, :indeps].as_matrix()
    X = df.iloc[:, indeps:].as_matrix()
    return reparse_data(X, Y, header, **kw)
