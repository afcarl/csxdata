import unittest

import numpy as np
from csxdata import CData, roots
from csxdata.utilities.parsers import parse_csv


etalonroot = roots["etalon"]


class TestTransformations(unittest.TestCase):
    """
    Dear Transformation Wrapper Classes,

    I would like you to:

    """
    def setUp(self):
        self.X_, self.y_, headers = parse_csv(etalonroot + "/input.csv")

        self.data = CData((self.X_, self.y_), cross_val=0)

    def test_standardization_on_etalon(self):
        self.data.reset_data(shuff=False)

        calcme = parse_csv(etalonroot + "std.csv", dtype="float64")[0]
        calcme = np.sort(calcme.ravel())

        self.data.transformation = "std"
        X = np.round(self.data.learning.astype("float64"), 3)
        X = np.sort(X.ravel())

        self.assertEqual(self.data.transformation, "std",
                         "The transformation property is faulty!")
        self.assertTrue(np.all(np.equal(X, calcme)), "Standardization is faulty!")

    def test_pca_on_etalon(self):
        self.data.reset_data(shuff=False)

        calcme = parse_csv(etalonroot + "pca.csv", dtype="float64")[0]
        calcme = np.round(np.sort(np.abs(calcme.ravel())), 1)

        self.data.transformation = "pca"
        X = self.data.learning.astype("float64")
        X = np.round(np.sort(np.abs(X.ravel())), 1)
        eq = np.isclose(X, calcme)

        self.assertEqual(self.data.transformation, "pca",
                         "The transformation property is faulty!")
        self.assertTrue(np.all(eq), "PCA is faulty!")

    def test_lda_on_etalon(self):
        self.data.reset_data(shuff=False)

        calcme = parse_csv(etalonroot + "lda.csv", dtype="float64")[0]
        calcme = np.round(np.sort(np.abs(calcme.ravel())), 1)

        self.data.transformation = "lda"
        X = self.data.learning.astype("float64")
        X = np.round(np.sort(np.abs(X.ravel())), 1)
        eq = np.isclose(X, calcme)

        self.assertEqual(self.data.transformation, "lda",
                         "The transformation property is faulty!")
        self.assertTrue(np.all(eq), "LDA is faulty!")

    def test_ica_on_etalon(self):
        self.data.reset_data(shuff=False)

        calcme = parse_csv(etalonroot + "ica.csv", dtype="float64")[0]
        calcme = np.round(np.sort(np.abs(calcme.ravel())), 1)

        self.data.transformation = "ica"
        X = self.data.learning.astype("float64")
        X = np.round(np.sort(np.abs(X.ravel())), 1)

        self.assertEqual(self.data.transformation, "ica",
                         "The transformation property is faulty!")
        self.assertTrue(np.allclose(X, calcme, rtol=1.e-4, atol=1.e-7), "ICA is faulty!")

    def test_autoencoding_on_etalon(self):
        self.data.reset_data(shuff=False)
        self.data.transformation = ("ae", 10)
        self.assertEqual(self.data.transformation, "autoencoding",
                         "Autoencoding failed on the <transformation> property assertion!")
        self.assertEqual(self.data.learning.shape, (10, 10),
                         "Autoencoding failed on the output shape test!")


class TestEmbedding(unittest.TestCase):
    """
    Dear Embedding Wrapper Classes,

    I would like you to:
    + create embeddings from categories
    ++ create OneHot embedding
    ++ create random embedding into n dimensions
    + transfrom any category label into the appropriate embedding
    - translate an embedding back to readable label or dummycode
    """

    def setUp(self):
        self.X_, self.y_, headers = parse_csv(etalonroot + "/input.csv")

        self.data = CData((self.X_, self.y_), cross_val=0)

    def test_embedding_then_reverting_to_onehot_doesnt_break_shapes(self):
        self.data.reset_data(shuff=False)
        self.data.crossval = 0
        self.data.embedding = 10
        self.assertEqual(self.data.embedding, "embedding",
                         "<embedding> setter is faulty! (got {})".format(self.data.embedding))
        X, y = self.data.table()
        self.assertEqual(y.shape, (10, 10),
                         "Embedding of independent variables went wrong! (got shape {})".format(y.shape))

        del self.data.embedding
        self.assertEqual(self.data.embedding, "onehot",
                         "<embedding> deleter is faulty! (got {})".format(self.data.embedding))
        X, y = self.data.table()
        self.assertEqual(y.shape, (10, 3),
                         "OneHot of independent variables went wrong! (got shape {})".format(y.shape))

if __name__ == '__main__':
    unittest.main()
