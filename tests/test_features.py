import unittest

import numpy as np

from csxdata import etalon


class TestTransformations(unittest.TestCase):
    """
    Dear Transformation Wrapper Classes,

    I would like you to:

    """
    def setUp(self):
        self.data = etalon()
        self.X_, self.y_ = self.data.table("learning")

    def test_standardization_on_etalon(self):
        self.data.reset_data(shuff=False, transform=)

        calcme = etalon("std.csv").X
        calcme = np.sort(calcme.ravel())

        self.data.transformation = "std"
        X = np.round(self.data.learning.astype("float64"), 3)
        X = np.sort(X.ravel())

        self.assertEqual(self.data.transformation, "std",
                         "The transformation property is faulty!")
        self.assertTrue(np.allclose(calcme, X, rtol=0.02, atol=0.01), "Standardization is faulty!")

    def test_pca_on_etalon(self):
        self.data.reset_data(shuff=False, transform=)

        calcme = etalon("pca.csv").X
        calcme = np.round(np.sort(np.abs(calcme.ravel())), 1)

        self.data.transformation = "pca"
        X = self.data.learning.astype("float64")
        X = np.round(np.sort(np.abs(X.ravel())), 1)
        eq = np.isclose(X, calcme)

        self.assertEqual(self.data.transformation, "pca",
                         "The transformation property is faulty!")
        self.assertTrue(np.all(eq), "PCA is faulty!")

    def test_lda_on_etalon(self):
        self.data.reset_data(shuff=False, transform=)

        calcme = etalon("lda.csv").X
        calcme = np.round(np.sort(np.abs(calcme.ravel())), 1)

        self.data.transformation = "lda"
        X = self.data.learning.astype("float64")
        X = np.round(np.sort(np.abs(X.ravel())), 1)
        eq = np.isclose(X, calcme)

        self.assertEqual(self.data.transformation, "lda",
                         "The transformation property is faulty!")
        self.assertTrue(np.all(eq), "LDA is faulty!")

    def test_ica_on_etalon(self):
        self.data.reset_data(shuff=False, transform=)

        calcme = etalon("ica.csv").X
        calcme = np.round(np.sort(np.abs(calcme.ravel())), 1)

        self.data.transformation = "ica"
        X = self.data.learning.astype("float64")
        X = np.round(np.sort(np.abs(X.ravel())), 1)

        self.assertEqual(self.data.transformation, "ica",
                         "The transformation property is faulty!")
        self.assertTrue(np.allclose(X, calcme, rtol=1.e-3, atol=1.e-5), "ICA is faulty!")

    def test_autoencoding_on_etalon(self):
        self.data.reset_data(shuff=False, transform=)
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
        self.data = etalon()
        self.X_, self.y_ = self.data.table("learning")

    def test_embedding_then_reverting_to_onehot_doesnt_break_shapes(self):
        self.data.reset_data(shuff=False, transform=)
        self.data.embedding = 10
        self.assertEqual(self.data.embedding, "embedding",
                         "<embedding> setter or getter is faulty! (got {})".format(self.data.embedding))
        X, Y = self.data.table()
        self.assertEqual(Y.shape, (10, 10),
                         "Embedding of independent variables went wrong! (got shape {})".format(Y.shape))

        del self.data.embedding
        self.assertEqual(self.data.embedding, "onehot",
                         "<embedding> deleter or getter is faulty! (got {})".format(self.data.embedding))
        X, Y = self.data.table()
        self.assertEqual(Y.shape, (10, 3),
                         "OneHot of independent variables went wrong! (got shape {})".format(Y.shape))

    def test_translate_with_onehot(self):
        self.data.reset_data(shuff=False, transform=)
        self.data.embedding = 0
        X, Y = self.data.table("learning", shuff=False)
        transl = self.data.translate(Y)
        for tr, y in zip(transl, self.data.Y):
            self.assertEqual(tr, y)

    def test_translate_with_embedding(self):
        self.data.reset_data(shuff=False, transform=)
        self.data.embedding = 2
        X, Y = self.data.table("learning", shuff=False)
        transl = self.data.translate(Y)
        for tr, y in zip(transl, self.data.Y):
            self.assertEqual(tr, y)


if __name__ == '__main__':
    unittest.main()
