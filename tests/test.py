"""
Dear CData,

I would like you to:
+ hold categorical data for me.
+ partition the data to learning and testing cases
- be able to generate weights based on the representation ratio of different classes
- transform (whiten, autoencode, standardize) the independent variables
 and adjust the <inputs_required> accordingly.
 These transformations should fitted only on the learning data!
- dummycode/embed the categorical variable:
 create the one-hot vector representations of categories OR
 embed the categorical variable into N-space,
 adjust <outputs_required> accordingly,
 and be able to translate the network output back to human readable class names
- be able to reset transformations and embeddings if this is desirable
 without the loss of information.
- create a learning table from the data
- generate random batches from the data
"""


import numpy as np
import unittest

from csxdata.frames import CData
from csxdata.const import roots
from csxdata.utilities.parsers import parse_csv


X_, y_, headers = parse_csv(roots["etalon"] + "input.csv")


data = CData((X_, y_), cross_val=0)
data.transformation = ("pca", 1)
data2 = CData((X_, y_), cross_val=0)


class TestCategorical(unittest.TestCase):

    def test_initialization(self):
        new_data = CData((X_, y_), cross_val=0.5)
        self.assertEqual(new_data.embedding, "onehot",
                         "<embedding> property is faulty after initialization!")
        self.assertEqual(len(new_data.categories), 3,
                         "Invalid determination of categories! (got {})".format(new_data.categories))
        self.assertEqual(new_data.crossval, 0.5,
                         "Wrong <crossval> value in data!")
        self.assertEqual(new_data.N, 5)
        self.assertEqual(new_data.N, new_data.learning.shape[0],
                         "Validation data splitting went wrong @ learning!")
        self.assertEqual(new_data.n_testing, 5)
        self.assertEqual(new_data.n_testing, new_data.testing.shape[0],
                         "Validation data splitting went wrong @ testing!")

    def test_reset(self):
        er = "Difference detected in data shapes"
        data.reset_data(shuff=False)
        self.assertEqual(data.learning.shape, (10, 3), msg=er)
        self.assertEqual(data.learning.shape, data2.learning.shape, msg=er)
        sm1, sm2 = np.sum(data.data), np.sum(data2.data)
        self.assertEqual(sm1, sm2,
                         msg="The sums of learning data differ by {}!\n{}\n{}"
                         .format(abs(sm1 - sm2), sm1, sm2))

    def test_writability(self):
        with self.assertRaises(ValueError):
            data.data[0][0] = 2.0

    def test_splitting(self):
        data.reset_data(shuff=False)
        data.crossval = 5
        self.assertEqual(data.crossval, 0.5,
                         "Wrong <crossval> value in data!")
        self.assertEqual(data.N, 5)
        self.assertEqual(data.N, data.learning.shape[0],
                         "Validation data splitting went wrong @ learning!")
        self.assertEqual(data.n_testing, 5)
        self.assertEqual(data.n_testing, data.testing.shape[0],
                         "Validation data splitting went wrong @ testing!")

    def test_weighing(self):
        pass  # TODO: implement

    def test_concatenate(self):
        newdata = CData((X_, y_), cross_val=0)
        data.reset_data(shuff=False)
        data.crossval = 0
        newdata.concatenate(data)
        self.assertEqual(newdata.N, 20, "Split after concatenation went wrong!")
        self.assertEqual(newdata.data.shape, (20, 3), "Shapes went haywire after concatenation!")


class TestTransformations(unittest.TestCase):

    def test_standardization(self):
        data.reset_data(shuff=False)

        calcme = parse_csv(roots["etalon"] + "std.csv", dtype="float64")[0]
        calcme = np.sort(calcme.ravel())

        data.transformation = "std"
        X = np.round(data.learning.astype("float64"), 3)
        X = np.sort(X.ravel())

        self.assertEqual(data.transformation, "standardization",
                         "The transformation property is faulty!")
        self.assertTrue(np.all(np.equal(X, calcme)), "Standardization is faulty!")

    def test_pca(self):
        data.reset_data(shuff=False)

        calcme = parse_csv(roots["etalon"] + "pca.csv", dtype="float64")[0]
        calcme = np.round(np.sort(np.abs(calcme.ravel())), 1)

        data.transformation = "pca"
        X = data.learning.astype("float64")
        X = np.round(np.sort(np.abs(X.ravel())), 1)
        eq = np.isclose(X, calcme)

        self.assertEqual(data.transformation, "pca",
                         "The transformation property is faulty!")
        self.assertTrue(np.all(eq), "PCA is faulty!")

    def test_autoencoding(self):
        data.reset_data(shuff=False)
        data.transformation = ("ae", 10)
        self.assertEqual(data.transformation, "autoencoding",
                         "Autoencoding failed on the <transformation> property assertion!")
        self.assertEqual(data.learning.shape, (10, 10),
                         "Autoencoding failed on the output shape test!")


class TestEmbedding(unittest.TestCase):

    def test_embedding_and_weighing(self):
        data.reset_data(shuff=False)
        data.crossval = 0
        data.embedding = 10
        self.assertEqual(data.embedding, "embedding",
                         "<embedding> setter is faulty! (got {})".format(data.embedding))
        X, y, w = data.table(weigh=True)
        self.assertEqual(y.shape, (10, 10),
                         "Embedding of independent variables went wrong! (got shape {})".format(y.shape))
        self.assertEqual(len([elem for elem in w if elem == 0.5]), 5,
                         "Weighing of samples went wrong after Embedding!")

        del data.embedding
        self.assertEqual(data.embedding, "onehot",
                         "<embedding> deleter is faulty! (got {})".format(data.embedding))
        X, y, w = data.table(weigh=True)
        self.assertEqual(y.shape, (10, 3),
                         "OneHot of independent variables went wrong! (got shape {})".format(y.shape))
        self.assertEqual(len([elem for elem in w if elem == 0.5]), 5,
                         "Weighing of samples went wrong after OneHot!")


if __name__ == '__main__':
    unittest.main()
