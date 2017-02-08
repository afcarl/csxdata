import unittest

import numpy as np
from csxdata import CData, roots
from csxdata.utilities.parsers import parse_csv

etalonroot = roots["etalon"]


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
        self.X_, self.y_, headers = parse_csv(etalonroot + "/input.csv")

        self.data = CData((self.X_, self.y_), cross_val=0)

    def test_initialization_on_etalon_with_given_parameters(self):
        new_data = CData((self.X_, self.y_), cross_val=0.5)
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
        data2 = CData((self.X_, self.y_), cross_val=0)
        er = "Difference detected in data shapes"

        self.data.crossval = 3
        self.data.embedding = 10
        self.data.transformation = ("pca", 1)

        self.data.reset_data(shuff=False)
        sm1, sm2 = np.sum(self.data.data), np.sum(data2.data)

        self.assertEqual(self.data.learning.shape, (7, 3), msg=er)
        self.assertEqual(sm1, sm2, msg="The sums of learning data differ by {}!\n{}\n{}"
                         .format(abs(sm1 - sm2), sm1, sm2))

    def test_core_data_is_readonly(self):
        with self.assertRaises(ValueError):
            self.data.data[0][0] = 2.0

    def test_setter_sets_crossval_getter_right(self):
        self.data.crossval = 5
        self.assertEqual(self.data.crossval, 0.5,
                         "Wrong <crossval> value in data!")
        self.assertEqual(self.data.N, 5)
        self.assertEqual(self.data.N, self.data.learning.shape[0],
                         "Validation data splitting went wrong @ learning!")
        self.assertEqual(self.data.n_testing, 5)
        self.assertEqual(self.data.n_testing, self.data.testing.shape[0],
                         "Validation data splitting went wrong @ testing!")

    def test_weights_sum_to_N(self):
        w = self.data.sample_weights
        self.assertEqual(round(w.sum()), self.data.N)

    def test_concatenate_produces_right_shape(self):
        newdata = CData((self.X_, self.y_), cross_val=0)
        newdata.concatenate(self.data)
        self.assertEqual(newdata.N, 20, "Split after concatenation went wrong!")
        self.assertEqual(newdata.data.shape, (20, 3), "Shapes went haywire after concatenation!")

    def test_batches_from_generator_are_shaped_and_distributed_right(self):
        i = 0
        self.data.crossval = 2
        self.data.transformation = ("ae", 15)
        self.data.embedding = 3
        for X, y, w in self.data.batchgen(2, weigh=True):
            self.assertEqual(X.shape, (2, 15))
            self.assertEqual(y.shape, (2, 3))
            self.assertEqual(w.shape, (2,))
            i += 1
        self.assertEqual(i, 4, msg="Number of batches differ. Got {} expected {}".format(i, 4))

    def test_neurons_required_property_on_untransformed_data(self):
        inshape, outputs = self.data.neurons_required
        self.assertIsInstance(inshape, tuple, "Input shape is not in a tuple!")
        self.assertIsInstance(inshape, tuple, "Output shape is not in a tuple!")
        self.assertEqual(inshape, (3,))
        self.assertEqual(outputs, (3,))

    def test_neurons_required_proprety_after_heavy_transformation_then_resetting(self):
        self.data.embedding = 10
        self.data.transformation = ("pca", 2)
        inshape, outputs = self.data.neurons_required
        self.assertIsInstance(inshape, tuple, "Input shape is not in a tuple!")
        self.assertIsInstance(outputs, tuple, "Output shape is not in a tuple!")
        self.assertEqual(inshape, (2,), "Wrong input shape after transformation/embedding!")
        self.assertEqual(outputs, (10,), "Wrong output shape after transformation/embedding!")
        self.data.reset_data(shuff=False)
        inshape, outputs = self.data.neurons_required
        self.assertIsInstance(inshape, tuple, "Input shape is not in a tuple!")
        self.assertIsInstance(outputs, tuple, "Output shape is not in a tuple!")
        self.assertEqual(inshape, (3,), "Wrong input shape after resetting!")
        self.assertEqual(outputs, (3,), "Wrong output shape after resetting!")


if __name__ == '__main__':
    unittest.main()
