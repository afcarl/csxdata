import numpy as np

from csxdata import CData

rng = np.random.RandomState(10191)


def pull_etalon_data():
    from csxdata.const import roots

    return CData(roots["etalon"], cross_val=0.1, header=1, sep="\t", end="\n")


def test_transformations():

    def test_splitting():
        assert data.crossval == 0.1, \
            "Wrong <crossval> value in data!"
        data.crossval = 2
        assert data.crossval == 0.2, \
            "Wrong <crossval> value in data!"
        assert data.N == data.learning.shape[0] == 8, \
            "Validation data splitting went wrong @ learning!"
        assert data.n_testing == data.testing.shape[0] == 8, \
            "Validation data splitting went wrong @ testing!"

    def test_standardization():
        data.set_transformation("std", features=None)

    data = pull_etalon_data()
