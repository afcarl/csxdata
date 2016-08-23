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

    def shape_tests():
        assert mnist.learning.shape == mnist2.learning.shape == (70000, 784), \
            "Difference detected in data shapes"
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

    def test_splitting():
        mnist.crossval = 7000
        assert mnist.crossval == 0.1, \
            "Wrong <crossval> value in data!"
        assert mnist.N == mnist.learning.shape[0] == (70000 - 7000), \
            "Validation data splitting went wrong @ learning!"
        assert mnist.n_testing == mnist.testing.shape[0] == 7000, \
            "Validation data splitting went wrong @ testing!"
        print("Test on data partitioning passed!")

    lt = mnist_tolearningtable(roots["misc"] + "mnist.pkl.gz", fold=False)
    mnist = CData(lt, cross_val=0)

    writability_test()

    mnist2 = CData(lt)

    shape_tests()
    sum_test()

    print("<<< CData: Every test passed! >>>")


if __name__ == '__main__':
    categorical()
