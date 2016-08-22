import numpy as np

from csxdata import CData
from csxdata.const import roots

rng = np.random.RandomState(10191)


def test_transformations():

    def test_standardization():
        data.transformation = "std"
        assert data.transformation == "standardization", \
            "The transformation property is faulty!"

    data = CData(roots["etalon"], cross_val=0.1, header=1, sep="\t", end="\n")
    assert data.transformation == "None", \
        "The transformation property is faulty!"

