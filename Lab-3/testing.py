# testing part for A2 ques in lab 4 with the codes of lab 3

import unittest
import pandas as pd

from A2 import label_encode, one_hot_encode
from A4 import minkowski_dist
from A5 import minkowski_distance
from A7 import dot_product, euclidean_norm
from A8 import find_mean, find_variance, find_std


class TestLab3Functions(unittest.TestCase):

    # Test Label Encoding
    def test_label_encode(self):
        data = pd.Series(["Red", "Blue", "Red", "Green"])

        result = label_encode(data)

        expected = [0, 1, 0, 2]

        self.assertEqual(result, expected)


    # Test One-Hot Encoding
    def test_one_hot_encode(self):
        data = pd.Series(["Red", "Blue", "Red"])

        result = one_hot_encode(data)

        expected = [
            [1, 0],
            [0, 1],
            [1, 0]
        ]

        self.assertEqual(result.values.tolist(), expected)


    # Test Minkowski Distance
    def test_minkowski_dist(self):
        v1 = [1, 2, 3]
        v2 = [4, 5, 6]

        result = minkowski_dist(v1, v2, 2)

        self.assertAlmostEqual(result, 5.196152423)


    # Test Minkowski Distance for A5
    def test_minkowski_distance(self):
        v1 = [1, 2, 3]
        v2 = [4, 5, 6]

        result = minkowski_distance(v1, v2, 2)

        self.assertAlmostEqual(result, 5.196152423)


    # Test Dot Product
    def test_dot_product(self):
        a = [1, 2, 3]
        b = [4, 5, 6]

        result = dot_product(a, b)

        self.assertEqual(result, 32)


    # Test Euclidean Norm
    def test_euclidean_norm(self):
        v = [3, 4]

        result = euclidean_norm(v)

        self.assertAlmostEqual(result, 5.0)


    # Test Mean
    def test_find_mean(self):
        data = [2, 4, 6, 8]

        result = find_mean(data)

        self.assertAlmostEqual(result, 5.0)


    # Test Variance
    def test_find_variance(self):
        data = [2, 4, 6, 8]

        result = find_variance(data)

        self.assertAlmostEqual(result, 5.0)


    # Test Standard Deviation
    def test_find_std(self):
        data = [2, 4, 6, 8]

        result = find_std(data)

        self.assertAlmostEqual(result, 2.236067977)


if __name__ == "__main__":
    unittest.main()
