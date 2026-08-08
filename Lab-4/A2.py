import unittest

from A1_05 import minkowski_distance
from A1_05 import minkowski_distance
from A1_09 import calc_std, calc_mean, calc_variance
from A1_07 import my_dot, my_norm


class TestLabFunctions(unittest.TestCase):

    # Test Minkowski Distance
    def test_minkowski_distance(self):
        v1 = [1, 2, 3]
        v2 = [4, 5, 6]

        result = minkowski_distance(v1, v2, 2)

        self.assertAlmostEqual(result, 5.196152423)


    # Test Dot Product
    def test_my_dot(self):
        a = [1, 2, 3]
        b = [4, 5, 6]

        result = my_dot(a, b)

        self.assertEqual(result, 32)


    # Test Euclidean Norm
    def test_my_norm(self):
        v = [3, 4]

        result = my_norm(v)

        self.assertAlmostEqual(result, 5.0)


    # Test Mean
    def test_calc_mean(self):
        data = [2, 4, 6, 8]

        result = calc_mean(data)

        self.assertAlmostEqual(result, 5.0)


    # Test Variance
    def test_calc_variance(self):
        data = [2, 4, 6, 8]

        result = calc_variance(data)

        self.assertAlmostEqual(result, 5.0)


    # Test Standard Deviation
    def test_calc_std(self):
        data = [2, 4, 6, 8]

        result = calc_std(data)

        self.assertAlmostEqual(result, 2.236067977)


if __name__ == "__main__":
    unittest.main()