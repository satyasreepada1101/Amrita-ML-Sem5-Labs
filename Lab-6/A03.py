import pandas as pd
import math
import time

from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("features.csv")

# Use the same classes as the attached programs
df = df[df["person_id"].isin(["A", "B"])]

# Features
X = df.drop(
    ["person_id", "image_name"],
    axis=1
)

# Target
y = df["person_id"]


# ============================================================
# 2. SAME TRAIN-TEST SPLIT FOR ALL THREE METHODS
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)


# ============================================================
# 3. MY KNN - A1.py
# ============================================================

def calculate_distance_my_knn(row1, row2):
    """
    Euclidean distance used in A1.py.
    """

    distance = 0

    for i in range(len(row1)):

        distance += (
            row1[i] - row2[i]
        ) ** 2

    return math.sqrt(distance)


def insertion_sort_my_knn(distances):
    """
    Insertion sort used in A1.py.
    """

    for i in range(1, len(distances)):

        key = distances[i]

        j = i - 1

        while (
            j >= 0
            and distances[j][1] > key[1]
        ):

            distances[j + 1] = distances[j]

            j -= 1

        distances[j + 1] = key

    return distances


def get_neighbors_my_knn(
        X_train,
        y_train,
        test_row,
        k):

    distances = []

    for i in range(len(X_train)):

        distance = calculate_distance_my_knn(
            X_train.iloc[i].values,
            test_row.values
        )

        distances.append(
            (
                y_train.iloc[i],
                distance
            )
        )

    distances = insertion_sort_my_knn(
        distances
    )

    return distances[:k]


def predict_class_my_knn(
        X_train,
        y_train,
        test_row,
        k):

    neighbors = get_neighbors_my_knn(
        X_train,
        y_train,
        test_row,
        k
    )

    labels = []

    for label, distance in neighbors:

        labels.append(label)

    count = Counter(labels)

    max_votes = max(
        count.values()
    )

    winners = []

    for label, votes in count.items():

        if votes == max_votes:

            winners.append(label)

    winners.sort()

    return winners[0]


def predict_my_knn(
        X_train,
        y_train,
        X_test,
        k):

    predictions = []

    for i in range(len(X_test)):

        prediction = predict_class_my_knn(
            X_train,
            y_train,
            X_test.iloc[i],
            k
        )

        predictions.append(prediction)

    return predictions


# ============================================================
# 4. GENAI KNN - A01_01.py
# ============================================================

def euclidean_distance_genai(
        sample1,
        sample2):

    total = 0

    for i in range(len(sample1)):

        difference = (
            sample1[i] - sample2[i]
        )

        total += (
            difference * difference
        )

    return math.sqrt(total)


def insertion_sort_genai(neighbors):

    for i in range(1, len(neighbors)):

        current = neighbors[i]

        j = i - 1

        while j >= 0:

            should_move = (
                neighbors[j][0] > current[0]
                or
                (
                    neighbors[j][0] == current[0]
                    and neighbors[j][1] > current[1]
                )
            )

            if should_move:

                neighbors[j + 1] = neighbors[j]

                j -= 1

            else:

                break

        neighbors[j + 1] = current

    return neighbors


def find_neighbors_genai(
        X_train,
        y_train,
        test_sample,
        k):

    neighbors = []

    for i in range(len(X_train)):

        training_sample = (
            X_train.iloc[i].tolist()
        )

        distance = euclidean_distance_genai(
            test_sample,
            training_sample
        )

        class_label = y_train.iloc[i]

        neighbors.append(
            [
                distance,
                i,
                class_label
            ]
        )

    neighbors = insertion_sort_genai(
        neighbors
    )

    return neighbors[:k]


def majority_voting_genai(
        neighbors):

    vote_count = {}

    distance_sum = {}

    distance_count = {}

    for neighbor in neighbors:

        distance = neighbor[0]

        class_label = neighbor[2]

        if class_label not in vote_count:

            vote_count[class_label] = 0

            distance_sum[class_label] = 0

            distance_count[class_label] = 0

        vote_count[class_label] += 1

        distance_sum[class_label] += distance

        distance_count[class_label] += 1

    maximum_votes = max(
        vote_count.values()
    )

    tied_classes = []

    for class_label in vote_count:

        if vote_count[class_label] == maximum_votes:

            tied_classes.append(class_label)

    if len(tied_classes) == 1:

        return tied_classes[0]

    # Tie-breaking using average distance
    best_class = tied_classes[0]

    best_average_distance = (
        distance_sum[best_class]
        /
        distance_count[best_class]
    )

    for class_label in tied_classes[1:]:

        average_distance = (
            distance_sum[class_label]
            /
            distance_count[class_label]
        )

        if average_distance < best_average_distance:

            best_class = class_label

            best_average_distance = average_distance

    return best_class


def predict_genai(
        X_train,
        y_train,
        X_test,
        k):

    predictions = []

    for i in range(len(X_test)):

        test_sample = (
            X_test.iloc[i].tolist()
        )

        neighbors = find_neighbors_genai(
            X_train,
            y_train,
            test_sample,
            k
        )

        prediction = majority_voting_genai(
            neighbors
        )

        predictions.append(prediction)

    return predictions


# ============================================================
# 5. PERFORMANCE MEASUREMENT FUNCTION
# ============================================================

def calculate_metrics(
        y_test,
        predictions):

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="binary",
        pos_label="B",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="binary",
        pos_label="B",
        zero_division=0
    )

    f_score = f1_score(
        y_test,
        predictions,
        average="binary",
        pos_label="B",
        zero_division=0
    )

    return (
        accuracy,
        precision,
        recall,
        f_score
    )


# ============================================================
# 6. RUN ONE METHOD 10 TIMES
# ============================================================

def measure_my_knn():

    k = 3

    start_time = time.perf_counter()

    for i in range(10):

        predictions = predict_my_knn(
            X_train,
            y_train,
            X_test,
            k
        )

    end_time = time.perf_counter()

    average_time = (
        end_time - start_time
    ) / 10

    metrics = calculate_metrics(
        y_test,
        predictions
    )

    return metrics, average_time


def measure_sklearn_knn():

    k = 3

    start_time = time.perf_counter()

    for i in range(10):

        model = KNeighborsClassifier(
            n_neighbors=k
        )

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

    end_time = time.perf_counter()

    average_time = (
        end_time - start_time
    ) / 10

    metrics = calculate_metrics(
        y_test,
        predictions
    )

    return metrics, average_time


def measure_genai_knn():

    k = 3

    start_time = time.perf_counter()

    for i in range(10):

        predictions = predict_genai(
            X_train,
            y_train,
            X_test,
            k
        )

    end_time = time.perf_counter()

    average_time = (
        end_time - start_time
    ) / 10

    metrics = calculate_metrics(
        y_test,
        predictions
    )

    return metrics, average_time


# ============================================================
# 7. RUN PERFORMANCE COMPARISON
# ============================================================

my_metrics, my_time = measure_my_knn()

sklearn_metrics, sklearn_time = (
    measure_sklearn_knn()
)

genai_metrics, genai_time = (
    measure_genai_knn()
)


# ============================================================
# 8. STORE RESULTS
# ============================================================

results = pd.DataFrame({

    "Implementation": [
        "My KNN (A1)",
        "Scikit-Learn KNN",
        "GenAI KNN (A01_01)"
    ],

    "Accuracy": [
        my_metrics[0],
        sklearn_metrics[0],
        genai_metrics[0]
    ],

    "Precision": [
        my_metrics[1],
        sklearn_metrics[1],
        genai_metrics[1]
    ],

    "Recall": [
        my_metrics[2],
        sklearn_metrics[2],
        genai_metrics[2]
    ],

    "F-score": [
        my_metrics[3],
        sklearn_metrics[3],
        genai_metrics[3]
    ],

    "Average Time (seconds)": [
        my_time,
        sklearn_time,
        genai_time
    ]
})


# ============================================================
# 9. DISPLAY REPORT TABLE
# ============================================================

print("\n")
print("=" * 85)
print("PERFORMANCE COMPARISON OF k-NN IMPLEMENTATIONS")
print("=" * 85)

print(
    results.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F-score": "{:.4f}".format,
            "Average Time (seconds)": "{:.6f}".format
        }
    )
)

print("=" * 85)

print("\nDataset split: 70% Training / 30% Testing")
print("Number of runs for timing: 10")
print("k value: 3")


# ============================================================
# 10. SAVE RESULTS AS CSV
# ============================================================

results.to_csv(
    "knn_performance_comparison.csv",
    index=False
)

print(
    "\nResults saved to "
    "knn_performance_comparison.csv"
)