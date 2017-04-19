"""Pure Python and Python stdlib based utilities are here.
This module aims to be PyPy compatible."""
import warnings


def euclidean(itr, target):
    import math
    assert len(itr) == len(target), "Can't perform distance calculation"
    res = math.sqrt(sum([(itr[i]-target[i])**2 for i in range(len(itr))]))
    return res


def chooseN(iterable: list, n=1):
    """Choose n elements randomly from an iterable and remove the element"""
    return [choose(iterable) for _ in range(n)]  # TODO: untested. Is <iterable> modified in-place?


def choose(iterable: list):
    """Chooses an element randomly from a list, then removes it from the list"""
    import random
    return iterable.pop(random.randrange(len(iterable)))  # TODO: untested. Is <iterable> modified in-place?


def feature_scale(iterable, from_=0, to=1):
    """Scales the elements of a vector between from_ and to uniformly"""
    # TODO: untested
    if type(iterable) not in (list, tuple):
        iterable = list(iterable)
    if max(iterable) + min(iterable) == 0:
        warnings.warn("Every value is 0 in iterable!", RuntimeWarning)
        return type(iterable)([from_ for _ in range(len(iterable))])

    out = []
    for e in iterable:
        try:
            x = ((e - min(iterable)) / (max(iterable) - min(iterable)) * (to - from_)) + from_
        except ZeroDivisionError:
            x = 0
        out.append(x)
    return out


def avg(iterable):
    return sum(iterable) / len(iterable)


def dehungarize(src, outflpath=None, incoding=None, outcoding=None, **reparse_kw):

    hun_to_asc = {"á": "a", "é": "e", "í": "i",
                  "ó": "o", "ö": "o", "ő": "o",
                  "ú": "u", "ü": "u", "ű": "u",
                  "Á": "A", "É": "E", "Í": "I",
                  "Ó": "O", "Ö": "O", "Ő": "O",
                  "Ú": "U", "Ü": "U", "Ű": "U"}

    if ("/" in src or "\\" in src) and len(src) < 200:
        src = pull_text(src, coding=incoding)
    src = "".join(char if char not in hun_to_asc else hun_to_asc[char] for char in src)
    if reparse_kw:
        src = reparse_txt(src, **reparse_kw)
    if outflpath is None:
        return src
    else:
        with open(outflpath, "w", encoding=outcoding) as outfl:
            outfl.write(src)
            outfl.close()


def ravel(a):
    """Recursive function to flatten a list of lists of lists..."""
    if not a:
        return a
    if isinstance(a[0], list):
        return ravel(a[0]) + ravel(a[1:])
    return a[:1] + ravel(a[1:])


def pull_text(src, coding="utf-8-sig", **reparse_kw):
    with open(src, mode="r", encoding=coding) as opensource:
        src = opensource.read()
    if reparse_kw:
        src = reparse_txt(src, **reparse_kw)
    return src


def reparse_txt(src, **kw):
    get = kw.get
    lower = get("lower", False)
    dehun = get("dehun", False)
    decimal = get("decimal", False)
    replace = get("replace", None)
    if dehun:
        src = dehungarize(src, decimal=decimal)
    if lower:
        src = src.lower()
    if replace is not None:
        # noinspection PyCallingNonCallable
        src = replace(src)
    return src


def isnumber(string: str):
    if not string:
        return False
    s = string[1:] if string[0] == "-" and "-" not in string[1:] else string
    if s.isdigit() or s.isnumeric():
        return True
    if "." not in s or s.count(".") > 1:
        return False
    if all(part.isdigit() for part in s.split(".")):
        return True
    return False
