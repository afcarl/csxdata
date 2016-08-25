import unittest

from csxdata.utilities.containers import Pipeline


class TestContainers(unittest.TestCase):
    def setUp(self):
        self.pipe = Pipeline(3)

    def test_pipe(self):
        a, b, c = "1st", "2nd", "3rd"

        for var in (a, b, c):
            self.assertIs(self.pipe.add_top(var), None, "Pipe returned something")
        self.assertEqual(self.pipe.free, 0)
        self.assertEqual(self.pipe.add_top(c), a)
        self.assertEqual(self.pipe.top, "3rd")
        self.assertEqual(self.pipe.bottom, "2nd")

if __name__ == '__main__':
    unittest.main()
