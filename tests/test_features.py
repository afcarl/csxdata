import numpy as np
rng = np.random.RandomState(10191)

def test_transformations():
    from csxdata import CData

    def test_standardization():

    etalonX = rng.randn(10, 3)
    etalony = rng.randint(0, 4, 10)

    data = CData((etalonX, etalony))
    data.set_transformation("standardization", features=None)
