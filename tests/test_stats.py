import unittest

import numpy as np

from csxdata.stats import skewkurt, sw, ks, ad


class TestNormalityTests(unittest.TestCase):
    def setUp(self):
        self.X = np.random.randn(1000, 1)

    def test_skewkurt(self):
        stat = skewkurt(self.X)[0]
        self.assertTrue(stat)

    def test_sw(self):
        stat = sw(self.X)[0]
        self.assertTrue(stat)

    def test_ks(self):
        stat = ks(self.X)[0]
        self.assertTrue(stat)

    def test_ad(self):
        stat = ad(self.X)[0]
        self.assertTrue(stat)

if __name__ == '__main__':
    unittest.main()
