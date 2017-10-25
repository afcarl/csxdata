from .frame.categorical import CData
from .frame.regression import RData
from .frame.eagertext import EagerText, LazyText, WordSequence
from .utilities.const import roots, log


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
    if not exists(roots["logs"] + "csxdata.log"):
        warnings.warn("Main logstring doesn't exist! Creating it...", RuntimeWarning)
        log("Created")
    # if not exists(roots["etalon"]):
    #     warnings.warn("Root folder for etalon data doesn't exist! Can't run tests this way...",
    #                   RuntimeWarning)

    if verbose:
        print("CsxData sanity check passed!")


sanity_check(0)


"""
TODO:
- data.data shouldn't be held in memory! It should be a generator expression.
or data.learning...
? So should be data.table()
- implement scaling as a feature!
- implement ICA and ZCA as a feature
? implement FFT as a feature?
"""
