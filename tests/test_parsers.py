import unittest
from csxdata.utilities.parsers import parse_text
from csxdata import roots


class TestParseText(unittest.TestCase):
    def setUp(self):
        self.path = roots["txt"] + "petofi.txt"

    def test_init(self):
        data, ngrams = parse_text(self.path, coding="utf-8-sig")
        self.assertEqual(len(ngrams), 92)

if __name__ == '__main__':
    unittest.main()
