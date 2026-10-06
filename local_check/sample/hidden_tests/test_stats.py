import unittest

from stats import average, median


class TestStats(unittest.TestCase):
    def test_average_empty(self):  # FAIL_TO_PASS: должен пройти после исправления
        self.assertEqual(average([]), 0.0)

    def test_average(self):  # PASS_TO_PASS: не должен сломаться
        self.assertEqual(average([1, 2, 3]), 2)

    def test_median(self):
        self.assertEqual(median([3, 1, 2]), 2)
        self.assertEqual(median([4, 1, 2, 3]), 2.5)


if __name__ == "__main__":
    unittest.main()
