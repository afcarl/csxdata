import unittest

from csxdata.utilities.pure import *


class TestPadnumberFunction(unittest.TestCase):

    def setUp(self):
        self.my_int, self.max_int = 12, 23400

    def test_with_integer(self):
        pad_int = padnumber(self.my_int, self.max_int)

        self.assertEqual(len(pad_int), len(str(self.max_int)))
        self.assertEqual(pad_int, "   12")

    def test_with_float(self):
        my_float, max_float = 12.2, 12.456

        pad_float = padnumber(my_float, max_float, before=False)

        self.assertEqual(len(pad_float), len(str(max_float)))
        self.assertEqual(pad_float, "12.2  ")
        
    def test_raises_when_len_of_max_is_smaller(self):
        my_int = 123456
        with self.assertRaises(ValueError):
            padnumber(my_int, self.max_int)


class TestNiceRound(unittest.TestCase):

    def setUp(self):
        self.big_ugly = 113232.2334
        self.normal_ugly = 3.1415
        self.small_ugly = 0.0012

    def test_with_small_float(self):
        small_nice = niceround(self.small_ugly, 2)

        self.assertEqual(small_nice, "0.00")

    def test_with_normal_float(self):
        normal_nice = niceround(self.normal_ugly, 2)

        self.assertEqual(normal_nice, "3.14")

    def test_wit_big_float(self):
        big_nice = niceround(self.big_ugly, 2)

        self.assertEqual(big_nice, "113232.23")


class TestMisc(unittest.TestCase):

    def test_recursive_list_flattening(self):
        disgusting_list = [[["asd", "dsa", "hell"], ["asd", ["hugs", "fire"]], []],
                           [[], ["eigenvalues", "eigenvectors", ["Harbingers", "Scythes"]]],
                           "World domination", "Conquest of Paradise"]
        nice_list = ravel(disgusting_list)
        self.assertEqual(len(nice_list), 12)


if __name__ == '__main__':
    unittest.main()
