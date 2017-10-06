import numpy as np

from .reparse import reparse_data, reparse_txt
from ..utilities.misc import isnumber, dehungarize
from ..utilities.vectorop import to_ngrams, to_wordarray


def array(A, indeps=1, headers=1, dtype="float32"):
    header = A[:headers].ravel() if headers else None
    matrix = A[headers:] if headers else A
    X, Y = np.split(matrix, indeps, axis=1)
    X[~np.vectorize(isnumber)(X)] = "nan"
    X = X.astype(dtype)
    return X, Y, header


def learningtable(source, dtype="float32"):
    X, y = source
    return X.astype(dtype), y, None


def txt(source, ngram=None, **kw):
    if ("\\" in source or "/" in source) and len(source) < 200:
        with open(source, mode="r", encoding=kw.pop("encoding", "utf-8")) as opensource:
            source = opensource.read()
    source = reparse_txt(source, **kw)
    return to_ngrams(np.array(list(source)), ngram) if ngram else to_wordarray(source)


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


def xlsx(source, indeps=1, headers=1, sheetname=0, skiprows=None, skip_footer=0, **kw):
    import pandas as pd
    df = pd.read_excel(source, sheetname, max(0, headers-1), skiprows, skip_footer)
    return dataframe(df, indeps, **kw)


def dataframe(df, indeps=1, **kw):
    header = df.columns
    if kw.pop("dropna", False):
        df.dropna(inplace=True)
    Y = df.iloc[:, :indeps].as_matrix().astype(str)
    X = df.iloc[:, indeps:].as_matrix()
    return reparse_data(X, Y, header, **kw)


def parse_source(source, indeps, headers, **kw):
    if isinstance(source, np.ndarray):
        return array(source, indeps, headers, kw.pop("floatX", "float32"))
    elif isinstance(source, tuple):
        return learningtable(source, kw.pop("floatX", "float32"))
    else:
        import pandas as pd
        if isinstance(source, pd.DataFrame):
            return dataframe(source, indeps, **kw)
    if not isinstance(source, str):
        raise TypeError(f"Unsupported source type: {type(source)}")
    if ".pkl.gz" in source.lower():
        return learningtable(source, kw.pop("floatX", "float32"))
    elif source.lower()[-4:] in (".csv", ".txt"):
        return csv(source, indeps, headers, **kw)
    elif source.lower()[-4:] in (".xls", "xlsx"):
        return xlsx(source, indeps, headers, **kw)
    else:
        raise ValueError(f"Unsupported source: {source}")
