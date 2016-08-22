import numpy as np

from csxdata import CData
from csxdata.const import roots
from csxdata.utilities.parsers import parse_csv


calcme = np.array([[1.033, -0.742,  2.662],
                   [-0.017, -1.087, -0.229],
                   [-0.847,  0.364,  0.194],
                   [1.102,  0.544, -0.749],
                   [-0.289, -0.787, -0.626],
                   [-1.737, -1.689, -0.137],
                   [0.314,  0.753, -0.084],
                   [1.333,  1.367, -1.339],
                   [-1.348, -0.111,  0.112],
                   [0.455,  1.386,  0.198]])

calcme = np.sort(calcme.ravel())


def test_transformations():
    def test_standardization():
        data.transformation = "std"
        assert data.transformation == "standardization", \
            "The transformation property is faulty!"
        X = np.round(data.learning.astype("float64"), 3)
        X = np.sort(X.ravel())
        assert np.all(np.equal(X, calcme)), "Standardization is faulty!"
        print("Test passed on standardization!")

    data = CData(parse_csv(roots["etalon"])[:2], cross_val=0)
    assert data.transformation == "None", \
        "The transformation property is faulty!"
    test_standardization()
    data.reset_data(shuff=False)

if __name__ == '__main__':
    test_transformations()
    print("<<< All tests passed! >>>")
