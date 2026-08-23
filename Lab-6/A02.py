import unittest
import pandas as pd
import sys

sys.path.append(r"C:\Satya Sreepada\B.Tech\SEM 5\Machine Learning\Weekly Labs\Lab 5")

from A1 import (
    encode_data,
    impute_missing_values,
    calculate_distance,
    bubble_sort,
    selection_sort,
    insertion_sort,
    get_neighbors,
    predict_class
)

from A2 import weighted_knn
from A7 import newKNN


# ============================================================
# TEST DATA
# ============================================================

# Small dataset used for testing
X_train = pd.DataFrame({
    "feature1": [1, 2, 3, 8],
    "feature2": [1, 2, 3, 8]
})

y_train = pd.Series(["A", "A", "B", "B"])

test_sample = pd.Series({
    "feature1": 2,
    "feature2": 2
})


# ============================================================
# 1. ENCODING TESTS
# ============================================================

class TestEncoding(unittest.TestCase):

    def test_encode_data(self):
        """
        Check that encode_data() returns a DataFrame.
        """

        data = pd.DataFrame({
            "color": ["red", "blue", "red"]
        })

        result = encode_data(data)

        self.assertIsInstance(result, pd.DataFrame)

        self.assertEqual(
            len(result),
            3
        )


# ============================================================
# 2. MISSING VALUE IMPUTATION TESTS
# ============================================================

class TestImputation(unittest.TestCase):

    def test_missing_value_is_filled(self):
        """
        Check that missing numerical values are filled.
        """

        data = pd.DataFrame({
            "value": [10, 20, None, 30]
        })

        result = impute_missing_values(data)

        # There should be no missing values
        self.assertFalse(
            result["value"].isnull().any()
        )

    def test_mean_is_used_for_missing_value(self):
        """
        Check that the old implementation uses the mean
        to fill a missing numerical value.
        """

        data = pd.DataFrame({
            "value": [10, 20, None, 30]
        })

        result = impute_missing_values(data)

        # Mean = (10 + 20 + 30) / 3 = 20
        self.assertEqual(
            result["value"].iloc[2],
            20
        )


# ============================================================
# 3. DISTANCE TESTS
# ============================================================

class TestDistance(unittest.TestCase):

    def test_distance(self):
        """
        Check Euclidean distance calculation.
        """

        row1 = [0, 0]

        row2 = [3, 4]

        distance = calculate_distance(
            row1,
            row2
        )

        # Expected distance = 5
        self.assertEqual(
            distance,
            5
        )

    def test_distance_same_points(self):
        """
        Distance between identical points should be zero.
        """

        row1 = [5, 10]

        row2 = [5, 10]

        distance = calculate_distance(
            row1,
            row2
        )

        self.assertEqual(
            distance,
            0
        )


# ============================================================
# 4. SORTING TESTS
# ============================================================

class TestSorting(unittest.TestCase):

    def setUp(self):
        """
        Create the same unsorted data before every test.
        """

        self.distances = [
            ("A", 5),
            ("B", 2),
            ("C", 8),
            ("D", 1)
        ]

    def test_bubble_sort(self):
        """
        Check Bubble Sort.
        """

        result = bubble_sort(
            self.distances.copy()
        )

        expected = [
            ("D", 1),
            ("B", 2),
            ("A", 5),
            ("C", 8)
        ]

        self.assertEqual(
            result,
            expected
        )

    def test_selection_sort(self):
        """
        Check Selection Sort.
        """

        result = selection_sort(
            self.distances.copy()
        )

        expected = [
            ("D", 1),
            ("B", 2),
            ("A", 5),
            ("C", 8)
        ]

        self.assertEqual(
            result,
            expected
        )

    def test_insertion_sort(self):
        """
        Check Insertion Sort.
        """

        result = insertion_sort(
            self.distances.copy()
        )

        expected = [
            ("D", 1),
            ("B", 2),
            ("A", 5),
            ("C", 8)
        ]

        self.assertEqual(
            result,
            expected
        )


# ============================================================
# 5. NEIGHBOR SELECTION TESTS
# ============================================================

class TestNeighbors(unittest.TestCase):

    def test_get_neighbors(self):
        """
        Check whether the k nearest neighbours are returned.
        """

        neighbors = get_neighbors(
            X_train,
            y_train,
            test_sample,
            3
        )

        # There should be exactly 3 neighbours
        self.assertEqual(
            len(neighbors),
            3
        )

    def test_nearest_neighbor(self):
        """
        Check that the closest sample is selected first.
        """

        neighbors = get_neighbors(
            X_train,
            y_train,
            test_sample,
            3
        )

        # First neighbour should be the closest
        self.assertEqual(
            neighbors[0][0],
            "A"
        )

        # Distance should be zero because
        # test_sample = [2, 2] and training sample [2, 2]
        self.assertEqual(
            neighbors[0][1],
            0
        )


# ============================================================
# 6. NORMAL VOTING TESTS
# ============================================================

class TestVoting(unittest.TestCase):

    def test_majority_voting(self):
        """
        Check whether majority voting selects the correct class.
        """

        # The closest 3 samples are:
        #
        # A
        # A
        # B
        #
        # Therefore prediction should be A.

        prediction = predict_class(
            X_train,
            y_train,
            test_sample,
            3
        )

        self.assertEqual(
            prediction,
            "A"
        )

    def test_voting_with_k_one(self):
        """
        With k=1, prediction should be the class
        of the nearest neighbour.
        """

        prediction = predict_class(
            X_train,
            y_train,
            test_sample,
            1
        )

        self.assertEqual(
            prediction,
            "A"
        )


# ============================================================
# 7. WEIGHTED kNN TESTS
# ============================================================

class TestWeightedKNN(unittest.TestCase):

    def test_weighted_knn(self):
        """
        Check whether weighted kNN returns a valid class.
        """

        prediction = weighted_knn(
            X_train,
            y_train,
            test_sample,
            3
        )

        self.assertIn(
            prediction,
            ["A", "B"]
        )

    def test_weighted_knn_zero_distance(self):
        """
        Check weighted kNN when a neighbour has zero distance.

        The old implementation gives a very large weight
        to a zero-distance neighbour.
        """

        prediction = weighted_knn(
            X_train,
            y_train,
            test_sample,
            1
        )

        self.assertEqual(
            prediction,
            "A"
        )


# ============================================================
# 8. KNN CLASS fit() TESTS
# ============================================================

class TestKNNFit(unittest.TestCase):

    def test_fit_stores_training_data(self):
        """
        Check whether fit() correctly stores X_train and y_train.
        """

        model = newKNN(k=3)

        model.fit(
            X_train,
            y_train
        )

        self.assertIsNotNone(
            model.X_train
        )

        self.assertIsNotNone(
            model.y_train
        )

        self.assertEqual(
            len(model.X_train),
            len(X_train)
        )

        self.assertEqual(
            len(model.y_train),
            len(y_train)
        )


# ============================================================
# 9. KNN CLASS predict() TESTS
# ============================================================

class TestKNNPredict(unittest.TestCase):

    def test_predict(self):
        """
        Check whether predict() returns predictions.
        """

        model = newKNN(k=3)

        model.fit(
            X_train,
            y_train
        )

        X_test = pd.DataFrame({
            "feature1": [2],
            "feature2": [2]
        })

        predictions = model.predict(
            X_test
        )

        self.assertEqual(
            len(predictions),
            1
        )

        self.assertEqual(
            predictions[0],
            "A"
        )

    def test_predict_multiple_samples(self):
        """
        Check prediction for multiple test samples.
        """

        model = newKNN(k=3)

        model.fit(
            X_train,
            y_train
        )

        X_test = pd.DataFrame({
            "feature1": [2, 8],
            "feature2": [2, 8]
        })

        predictions = model.predict(
            X_test
        )

        self.assertEqual(
            len(predictions),
            2
        )

        self.assertIn(
            predictions[0],
            ["A", "B"]
        )

        self.assertIn(
            predictions[1],
            ["A", "B"]
        )


# ============================================================
# 10. KNN CLASS score() TESTS
# ============================================================

class TestKNNScore(unittest.TestCase):

    def test_score(self):
        """
        Check whether score() returns an accuracy value.
        """

        model = newKNN(k=3)

        model.fit(
            X_train,
            y_train
        )

        X_test = pd.DataFrame({
            "feature1": [2, 8],
            "feature2": [2, 8]
        })

        y_test = pd.Series([
            "A",
            "B"
        ])

        accuracy = model.score(
            X_test,
            y_test
        )

        self.assertEqual(
            accuracy,
            1.0
        )

    def test_score_range(self):
        """
        Accuracy should always be between 0 and 1.
        """

        model = newKNN(k=3)

        model.fit(
            X_train,
            y_train
        )

        X_test = pd.DataFrame({
            "feature1": [2],
            "feature2": [2]
        })

        y_test = pd.Series(["A"])

        accuracy = model.score(
            X_test,
            y_test
        )

        self.assertGreaterEqual(
            accuracy,
            0
        )

        self.assertLessEqual(
            accuracy,
            1
        )


# ============================================================
# RUN ALL TESTS
# ============================================================

if __name__ == "__main__":
    unittest.main()