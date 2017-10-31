import unittest

from csxdata.utilities.misc import *


class TestMisc(unittest.TestCase):

    def test_recursive_list_flattening(self):
        disgusting_list = [[["asd", "dsa", "hell"], ["asd", ["hugs", "fire"]], []],
                           [[], ["eigenvalues", "eigenvectors", ["Harbingers", "Scythes"]]],
                           "World domination", "Conquest of Paradise"]
        nice_list = ravel(disgusting_list)
        self.assertEqual(len(nice_list), 12)


if __name__ == '__main__':
    unittest.main()
