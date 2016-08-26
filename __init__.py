from .frames import CData, RData, Sequence
from .const import roots, log


def sanity_check(verbose=1):
    import os
    import warnings

    from os.path import exists

    if not exists(roots["data"]):
        warnings.warn("Data root doesn't exist!", RuntimeWarning)
    if not exists(roots["cache"]):
        warnings.warn("Cache directory doesn't exist! Creating it...", RuntimeWarning)
        os.mkdir(roots["cache"])
    if not exists(roots["logs"]):
        warnings.warn("Logs directory doesn't exist! Creating it...", RuntimeWarning)
        os.mkdir(roots["logs"])
    if not exists(roots["logs"] + ".csxdata.log"):
        warnings.warn("Main log doesn't exist! Creating it...", RuntimeWarning)
        log("Created")

    if verbose:
        print("CsxData sanity check passed!")


def get_etalon():
    from .utilities.parsers import parse_csv
    return CData(parse_csv(roots["etalon"] + "input.csv")[:2])


sanity_check(0)
etalon = get_etalon()

"""
TODO:
- data.data shouldn't be held in memory! It should be a generator expression.
or data.learning...
? So should be data.table()
- implement scaling as a feature!
- implement ICA and ZCA as a feature
"""