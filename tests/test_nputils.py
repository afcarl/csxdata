import unittest

from csxdata.utilities.nputils import *


class TestEuclidean(unittest.TestCase):

    def test_euclidean_vector_vs_vector_1(self):
        x1 = np.zeros((2,)).astype(float)
        x2 = np.ones((2,)).astype(float)
        y = np.sqrt(2)
        output = euclidean(x1, x2)
        self.assertEqual(output, y, "Test failed @ euclidean of vectors!")

    def test_euclidean_vector_vs_vector_2(self):
        x1 = np.array([15.1, 0.5, 13.45, 0.0, 187.0, 27.0, 18.0, 254.0, 0.8, 7.2])
        x2 = np.array([11.6258517, 4.04255166, 3.51548475, 1.66430278, 266.139903, 146.10648500000002,
                       111.96102, 18.085486500000002, 15.335202500000001, 5.7048872])
        y = 292
        output = int(euclidean(x1, x2))
        self.assertEqual(output, y, "Test fauiled @ euclideon of vectors #2!")

    def test_euclidean_vector_vs_matrix(self):
        x1 = np.zeros((2, 2)).astype(float)
        x2 = np.ones((2, 2)).astype(float)
        y = np.sqrt(2) * 2
        output = euclidean(x1, x2).sum()
        self.assertEqual(output, y, "Test failed @ euclidean of matrices!")

    def test_avgpool(self):
        pass  # TODO: implement


class TestCombination(unittest.TestCase):

    def test_vector_times_scalar(self):
        x = np.arange(10)
        w = 2
        y = np.arange(0, 20, 2)
        output = neuron(x, w, 0.0)
        self.assertTrue(np.all(np.equal(y, output)), "Test failed @ combination of vector with scalar!")

    def test_vector_times_vector(self):
        x = np.ones((10,)) * 2
        w = np.arange(10)
        y = float(np.arange(0, 20, 2).sum())
        output = neuron(x, w, 0.0)
        self.assertEqual(output, y, "Test failed @ combination of vector with vector!")

    def test_matrix_times_matrix(self):
        x = np.arange(12).reshape(3, 4)
        w = np.arange(16).reshape(4, 4)
        y = np.dot(x, w)
        output = neuron(x, w, 0.0)
        self.assertTrue(np.all(np.equal(y, output)), "Test failed @ combination of matrix with matrix!")


class TestTheRest(unittest.TestCase):

    def test_featscale(self):
        x = np.arange(3 * 4).reshape((3, 4)).astype(float)
        y = np.array([[0.0, 0.0, 0.0, 0.0],
                      [1.0, 1.0, 1.0, 1.0],
                      [2.0, 2.0, 2.0, 2.0]])
        output = featscale(x, ufctr=(0, 2))
        self.assertTrue(np.all(np.equal(y, output)), "Feature scale test failed!")

    def test_ravel_to_matrix(self):
        x = np.arange(2 * 3 * 4 * 5 * 6).reshape((2, 3, 4, 5, 6))
        yshape = (2, 3 * 4 * 5 * 6)
        output = ravel_to_matrix(x).shape
        self.assertTrue(np.all(np.equal(yshape, output)), "Test failed @ ravel_to_matrix!")


class TestShuffle(unittest.TestCase):

    def setUp(self):
        self.x = np.arange(10)
        self.y = list("ABCDEFGHIJ")

    def test_argshuffle(self):
        arg = argshuffle((self.x, np.array(self.y)))
        self.assertTrue(np.all(np.equal(np.sort(arg), self.x)))

    def test_shuffle(self):
        shx, shy = shuffle((self.x, np.array(self.y)))
        sharg = np.vectorize(lambda char: self.y.index(char))(shy)
        self.assertTrue(np.all(np.equal(sharg, shx)))
        self.assertTrue(np.all(np.equal(np.sort(sharg), self.x)))


class TestSumSort(unittest.TestCase):

    def setUp(self):
        self.x = np.array([3, 0, 4, 1, 2] * 5).reshape(5, 5)
        self.y = np.array([0, 1, 2, 3, 4] * 5).reshape(5, 5)
        self.arg = [1, 3, 4, 0, 2]

    def test_argsumsort(self):
        arg = argsumsort(self.x, axis=0)
        self.assertTrue(np.all(np.equal(arg, self.arg)))

    def test_sumsort(self):
        y = sumsort(self.x, axis=0)
        self.assertTrue(np.all(np.equal(y, self.y)))


if __name__ == '__main__':
    unittest.main()
