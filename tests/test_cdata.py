"""
Dear Categorical Dataframe,

I would like you to:
+ hold categorical data for me.
+ partition the data to learning and testing cases
+ be able to generate weights based on the representation ratio of different classes
+ transform (whiten, autoencode, standardize) the independent variables
 and adjust the <inputs_required> accordingly.
 These transformations should fitted only on the learning data!
+ dummycode/embed the categorical variable:
 create the one-hot vector representations of categories OR
 embed the categorical variable into N-space,
 adjust <outputs_required> accordingly,
 and be able to translate the network output back to human readable class names
+ be able to reset transformations and embeddings if this is desirable
 without the loss of information.
+ create a learning table from the data
+ generate random batches from the data
- Handle multiple labels and be able to average similarily labelled samples
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


class TestCData(unittest.TestCase):
    """
    Dear Categorical Dataframe,

    I would like you to:
    + hold categorical data for me.
    + partition the data to learning and testing cases
    + be able to generate weights based on the representation ratio of different classes
    + transform (whiten, autoencode, standardize) the independent variables
     and adjust the <inputs_required> accordingly.
     These transformations should fitted only on the learning data!
    + dummycode/embed the categorical variable:
     create the one-hot vector representations of categories OR
     embed the categorical variable into N-space,
     adjust <outputs_required> accordingly,
     and be able to translate the network output back to human readable class names
    + be able to reset transformations and embeddings if this is desirable
     without the loss of information.
    + create a learning table from the data
    + generate random batches from the data
    - Handle multiple labels and be able to average similarily labelled samples
    """

    def setUp(self):
        data.reset_data(shuff=False)
        data.crossval = 0

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
        data.crossval = 3
        data.embedding = 10
        data.transformation = ("pca", 1)
        data.reset_data(shuff=False)
        data.crossval = 0
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
        w = data.sample_weights
        self.assertEqual(round(w.sum()), data.N)

    def test_concatenate(self):
        newdata = CData((X_, y_), cross_val=0)
        newdata.concatenate(data)
        self.assertEqual(newdata.N, 20, "Split after concatenation went wrong!")
        self.assertEqual(newdata.data.shape, (20, 3), "Shapes went haywire after concatenation!")

    def test_batch_generator(self):
        i = 0
        data.crossval = 2
        data.transformation = ("ae", 15)
        data.embedding = 3
        for X, y, w in data.batchgen(2, weigh=True):
            self.assertEqual(X.shape, (2, 15))
            self.assertEqual(y.shape, (2, 3))
            self.assertEqual(w.shape, (2,))
            i += 1
        self.assertEqual(i, 4, msg="Number of batches differ. Got {} expected {}".format(i, 4))

    def test_inputs_outputs_vanilla(self):
        inshape, outputs = data.neurons_required
        self.assertIsInstance(inshape, int, "<Fanin> input shape is in a tuple!")
        self.assertEqual(inshape, 3)
        self.assertEqual(outputs, 3)

    def test_inputs_outputs_ultimate(self):
        data.embedding = 10
        data.transformation = ("pca", 2)
        inshape, outputs = data.neurons_required
        self.assertIsInstance(inshape, int, "<Fanin> input shape is in a tuple!")
        self.assertEqual(inshape, 2, "Wrong input shape after transformation/embedding!")
        self.assertEqual(outputs, 10, "Wrong output shape after transformation/embedding!")
        data.reset_data(shuff=False)
        inshape, outputs = data.neurons_required
        self.assertIsInstance(inshape, int, "<Fanin> input shape is in a tuple!")
        self.assertEqual(inshape, 3, "Wrong input shape after resetting!")
        self.assertEqual(outputs, 3, "Wrong output shape after resetting!")


class TestTransformations(unittest.TestCase):
    """
    Dear Transformation Wrapper Classes,

    I would like you to:

    """

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
    """
    Dear Embedding Wrapper Classes,

    I would like you to:
    + create embeddings from categories
    ++ create OneHot embedding
    ++ create random embedding into n dimensions
    + transfrom any category label into the appropriate embedding
    - translate an embedding back to readable label or dummycode
    """

    def test_embedding(self):
        data.reset_data(shuff=False)
        data.crossval = 0
        data.embedding = 10
        self.assertEqual(data.embedding, "embedding",
                         "<embedding> setter is faulty! (got {})".format(data.embedding))
        X, y = data.table()
        self.assertEqual(y.shape, (10, 10),
                         "Embedding of independent variables went wrong! (got shape {})".format(y.shape))

        del data.embedding
        self.assertEqual(data.embedding, "onehot",
                         "<embedding> deleter is faulty! (got {})".format(data.embedding))
        X, y = data.table()
        self.assertEqual(y.shape, (10, 3),
                         "OneHot of independent variables went wrong! (got shape {})".format(y.shape))


if __name__ == '__main__':
    unittest.main()
