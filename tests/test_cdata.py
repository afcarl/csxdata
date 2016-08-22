"""
Dear CData,

I would like you to:
- hold categorical data for me.
- partition the data to learning and testing cases
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


def categorical():
    from csxdata.frames import CData
    from csxdata.const import roots
    from csxdata.utilities.parsers import mnist_tolearningtable

    def test_embeddings():
        mnist.embedding = 4
        X, y = mnist.table("learning")
        assert y.shape[-1] == 4, "Embedding went wrong!"
        assert mnist.neurons_required[1] == 4, "<neurons_required> property failed after appliing Embed!"
        mnist.embedding = 0
        X, y = mnist.table("learning")
        assert y.shape[-1] == 10, "Resetting to OneHot went wrong!"
        assert mnist.neurons_required[1] == 10, "<neurons_required> property failed after appliing OneHot!"

    def test_autoencoding():
        mnist.set_transformation("ae", features=60)
        assert mnist.learning.shape[-1] == mnist.testing.shape[-1] == 60, \
            "Unsuccessful autoencoding!\nlearning shape:\t{}\ntesting shape:\t{}".format(mnist.learning.shape,
                                                                                         mnist.testing.shape)
        print("Autoencoding was successful! Test passed!")
        mnist.reset_data(shuff=False, transform=None, trparam=None)

    def shape_tests():
        assert all([i == j for i, j in zip(mnist.learning.shape, mnist2.learning.shape)]), \
            "Difference detected in data shapes"
        assert mnist.data.shape[-1] == 784, \
            "MNIST data lost its shape?"
        print("Shape tests passed on MNIST data!")

    def sum_test():
        sm1, sm2 = np.sum(mnist.data), np.sum(mnist2.data)
        assert sm1 == sm2, \
            "The sums of learning data differ by {}!\n{}\n{}".format(abs(sm1 - sm2), sm1, sm2)
        print("Test of summation passed!")

    def writability_test():
        assert np.sum(mnist.data - lt[0]) == 0, \
            "Data core remained unchanged! Test of subtraction passed!"
        try:
            mnist.data[0][0] = 2.0
        except ValueError:
            print("CData.data is read only. Test passed!")
        try:
            mnist.indeps /= 3
        except ValueError:
            print("Cdata.indeps is read only. Test passed!")

    def test_splitting(data):
        data.crossval = 2
        assert data.crossval == 0.2, \
            "Wrong <crossval> value in data!"
        assert data.N == data.learning.shape[0] == 8, \
            "Validation data splitting went wrong @ learning!"
        assert data.n_testing == data.testing.shape[0] == 2, \
            "Validation data splitting went wrong @ testing!"
        print("Test on data partitioning passed!")

    lt = mnist_tolearningtable(roots["misc"] + "mnist.pkl.gz", fold=False)
    mnist = CData(lt, cross_val=.1)

    writability_test()
    test_embeddings()
    test_autoencoding()

    mnist2 = CData(lt)

    shape_tests()
    sum_test()

    print("<<< CData: Every test passed! >>>")


if __name__ == '__main__':
    categorical()
