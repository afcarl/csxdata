import unittest
from csxdata import roots


class TestParseText(unittest.TestCase):
    def setUp(self):
        self.path = roots["txt"] + "petofi.txt"

if __name__ == '__main__':
    unittest.main()
