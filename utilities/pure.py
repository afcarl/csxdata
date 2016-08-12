"""Pure Python and Python stdlib based utilities are here.
This module aims to be PyPy compatible."""


def euclidean(itr, target):
    import math
    assert len(itr) == len(target), "Can't perform distance calculation"
    res = math.sqrt(sum([(itr[i]-target[i])**2 for i in range(len(itr))]))
    return res


def chooseN(iterable: list, N=1):
    """Choose N elements randomly from an iterable and remove the element"""
    return [choose(iterable) for _ in range(N)]  # TODO: untested. Is <iterable> modified in-place?


def choose(iterable: list):
    """Chooses an element randomly from a list, then removes it from the list"""
    import random
    out = random.choice(iterable)
    iterable.remove(out)
    return out  # TODO: untested. Is <iterable> modified in-place?


def feature_scale(iterable, from_=0, to=1):
    """Scales the elements of a vector between from_ and to uniformly"""
    # TODO: untested
    if max(iterable) + min(iterable) == 0:
        # print("Feature scale warning: every value is 0 in iterable!")
        return type(iterable)([from_ for _ in range(len(iterable))])

    out = []
    for e in iterable:
        try:
            x = ((e - min(iterable)) / (max(iterable) - min(iterable)) * (to - from_)) + from_
        except ZeroDivisionError:
            x = 0
        out.append(x)
    return type(iterable)(out)


def avg(iterable):
    return sum(iterable) / len(iterable)
