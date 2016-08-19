from .frames import CData, RData


def sanity_check():
    import os
    import warnings

    from os.path import exists
    from const import roots

    if not exists(roots["data"]):
        warnings.warn("Data root doesn't exist!", RuntimeWarning)
    if not exists(roots["cache"]):
        os.mkdir(roots["cache"])
    if not exists(roots["logs"]):
        os.mkdir(roots["logs"])
