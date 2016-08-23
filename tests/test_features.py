import numpy as np

from csxdata import CData
from csxdata.const import roots
from csxdata.utilities.parsers import parse_csv


def test_transformations():

    def test_standardization():
        calcme = parse_csv(roots["etalon"] + "std.csv", dtype="float64")[0]
        calcme = np.sort(calcme.ravel())

        data.transformation = "std"
        assert data.transformation == "standardization", \
            "The transformation property is faulty!"

        X = np.round(data.learning.astype("float64"), 3)
        X = np.sort(X.ravel())

        assert np.all(np.equal(X, calcme)), "Standardization is faulty!"
        print("Test passed on standardization!")

    def test_pca():
        calcme = parse_csv(roots["etalon"] + "pca.csv", dtype="float64")[0]
        calcme = np.round(np.sort(np.abs(calcme.ravel())), 1)

        data.transformation = "pca"
        assert data.transformation == "pca"

        X = data.learning.astype("float64")
        X = np.round(np.sort(np.abs(X.ravel())), 1)

        eq = np.isclose(X, calcme)

        assert np.all(eq), "PCA is faulty!"
        print("Test passed on PCA!")

    def test_autoencoding():
        data.transformation = ("ae", 10)
        assert data.transformation == "autoencoding",\
            "Autoencoding failed on the <transformation> property assertion!"
        assert data.learning.shape == (10, 10),\
            "Autoencoding failed on the output shape test!"
        print("Autoencoding passed!")

    data = CData(parse_csv(roots["etalon"] + "input.csv")[:2], cross_val=0)
    assert data.transformation == "None", \
        "The transformation property is faulty!"
    test_standardization()
    data.reset_data(shuff=False)
    test_pca()
    data.reset_data(shuff=False)
    test_autoencoding()
    print("<<< All tests passed on transformations! >>>")


def test_embeddings():

    def test_embedding_and_weighing():
        data.embedding = 10
        assert data.embedding == "embedding", "<embedding> setter is faulty! (got {})".format(data.embedding)
        X, y, w = data.table(weigh=True)
        assert y.shape == (10, 10), "Embedding of independent variables went wrong! (got shape {})".format(y.shape)
        assert len([elem for elem in w if elem == 0.5]) == 5, "Weighing of samples went wrong after Embedding!"
        print("Passed test on Embedding via property setter!")
        del data.embedding
        assert data.embedding == "onehot", "<embedding> deleter is faulty! (got {})".format(data.embedding)
        X, y, w = data.table(weigh=True)
        assert y.shape == (10, 3), "OneHot of independent variables went wrong! (got shape {})".format(y.shape)
        assert len([elem for elem in w if elem == 0.5]) == 5, "Weighing of samples went wrong after OneHot!"
        print("Passed test on OneHot via property deleter!")
        print("Embedding tests passed!")

    data = CData(parse_csv(roots["etalon"] + "input.csv")[:2], cross_val=0)
    assert data.embedding == "onehot", "<embedding> property is faulty! (got {})".format(data.embedding)
    print("Passed test in OneHot after initialization!")
    assert len(data.categories) == 3, "Invalid determination of categories! (got {})".format(data.categories)
    assert max(data.dummycode("learning")) == 2, "Invalid determination of dummycodes!"
    print("Passed tests of category determination and dummycoding!")
    test_embedding_and_weighing()
    print("<<< All tests passed on embeddings! >>>")


if __name__ == '__main__':
    test_transformations()
    test_embeddings()
    print("<<< All tests passed! >>>")

"""
TODO:
write test on embeddings which assert the back-translation.
Figure out some way to test this with embedding as well...
"""