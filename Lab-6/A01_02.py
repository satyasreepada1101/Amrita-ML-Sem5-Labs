import pandas as pd
import math

def label_encode_column(data, column_name):
    """
    Convert categorical values into numerical values
    using Label Encoding.
    """

    unique_values = data[column_name].dropna().unique()

    encoding = {}

    for i in range(len(unique_values)):
        encoding[unique_values[i]] = i

    data[column_name] = data[column_name].map(encoding)

    return data


def encode_categorical_columns(data, target_column):
    """
    Detect categorical columns and convert them into
    numerical values.
    """

    data = data.copy()

    for column in data.columns:

        if data[column].dtype == "object":

            data = label_encode_column(data, column)

    return data


def check_outliers(column):
    """
    Check whether a numerical column contains outliers
    using the IQR method.
    """

    q1 = column.quantile(0.25)
    q3 = column.quantile(0.75)

    iqr = q3 - q1

    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr

    outliers = column[
        (column < lower_limit) |
        (column > upper_limit)
    ]

    return len(outliers) > 0


def is_normal_distribution(column):
    """
    Check whether numerical data is approximately normally
    distributed using skewness.
    """

    skewness = column.skew()

    if -0.5 <= skewness <= 0.5:
        return True
    else:
        return False


def impute_missing_values(data):
    """
    Handle missing values.

    Numerical columns:
        Mean -> approximately normal data
        Median -> outliers or non-normal data

    Categorical columns:
        Mode
    """

    data = data.copy()

    for column in data.columns:

        if data[column].isnull().sum() > 0:

            if pd.api.types.is_numeric_dtype(data[column]):

                if check_outliers(data[column]):

                    value = data[column].median()
                    data[column] = data[column].fillna(value)

                elif is_normal_distribution(data[column]):

                    value = data[column].mean()
                    data[column] = data[column].fillna(value)

                else:

                    value = data[column].median()
                    data[column] = data[column].fillna(value)

            else:

                mode_value = data[column].mode()[0]
                data[column] = data[column].fillna(mode_value)

    return data


def euclidean_distance(sample1, sample2):
    """
    Calculate Euclidean distance.
    """

    total = 0

    for i in range(len(sample1)):

        difference = sample1[i] - sample2[i]

        total = total + (difference * difference)

    return math.sqrt(total)


def manhattan_distance(sample1, sample2):
    """
    Calculate Manhattan distance.
    """

    total = 0

    for i in range(len(sample1)):

        difference = abs(sample1[i] - sample2[i])

        total = total + difference

    return total


def minkowski_distance(sample1, sample2, p=3):
    """
    Calculate Minkowski distance.
    """

    total = 0

    for i in range(len(sample1)):

        difference = abs(sample1[i] - sample2[i])

        total = total + (difference ** p)

    return total ** (1 / p)


def calculate_distance(sample1, sample2, metric="euclidean"):
    """
    Select the required distance metric.
    """

    if metric == "euclidean":

        return euclidean_distance(sample1, sample2)

    elif metric == "manhattan":

        return manhattan_distance(sample1, sample2)

    elif metric == "minkowski":

        return minkowski_distance(sample1, sample2)

    else:

        print("Invalid distance metric.")

        return None


def bubble_sort(neighbors):
    """
    Sort neighbors using Bubble Sort.
    """

    n = len(neighbors)

    for i in range(n):

        for j in range(0, n - i - 1):

            if (
                neighbors[j][0] > neighbors[j + 1][0]
                or
                (
                    neighbors[j][0] == neighbors[j + 1][0]
                    and neighbors[j][1] > neighbors[j + 1][1]
                )
            ):

                temp = neighbors[j]

                neighbors[j] = neighbors[j + 1]

                neighbors[j + 1] = temp

    return neighbors


def selection_sort(neighbors):
    """
    Sort neighbors using Selection Sort.
    """

    n = len(neighbors)

    for i in range(n):

        smallest = i

        for j in range(i + 1, n):

            if (
                neighbors[j][0] < neighbors[smallest][0]
                or
                (
                    neighbors[j][0] == neighbors[smallest][0]
                    and neighbors[j][1] < neighbors[smallest][1]
                )
            ):

                smallest = j

        temp = neighbors[i]

        neighbors[i] = neighbors[smallest]

        neighbors[smallest] = temp

    return neighbors


def insertion_sort(neighbors):
    """
    Sort neighbors using Insertion Sort.
    """

    for i in range(1, len(neighbors)):

        current = neighbors[i]

        j = i - 1

        while j >= 0:

            if (
                neighbors[j][0] > current[0]
                or
                (
                    neighbors[j][0] == current[0]
                    and neighbors[j][1] > current[1]
                )
            ):

                neighbors[j + 1] = neighbors[j]

                j = j - 1

            else:

                break

        neighbors[j + 1] = current

    return neighbors


def sort_neighbors(neighbors, sorting_method="bubble"):
    """
    Select the sorting algorithm.
    """

    if sorting_method == "bubble":

        return bubble_sort(neighbors)

    elif sorting_method == "selection":

        return selection_sort(neighbors)

    elif sorting_method == "insertion":

        return insertion_sort(neighbors)

    else:

        print("Invalid sorting method.")

        return neighbors


def find_nearest_neighbors(
        features,
        target,
        test_sample,
        k,
        distance_metric="euclidean",
        sorting_method="bubble"):
    """
    Calculate distances and find k nearest neighbors.

    Tie-breaking:
    If two neighbors have the same distance,
    the smaller sample index is selected first.
    """

    neighbors = []

    for i in range(len(features)):

        training_sample = features.iloc[i].tolist()

        distance = calculate_distance(
            test_sample,
            training_sample,
            distance_metric
        )

        class_label = target.iloc[i]

        neighbors.append([
            distance,
            i,
            class_label
        ])

    neighbors = sort_neighbors(
        neighbors,
        sorting_method
    )

    nearest_neighbors = neighbors[:k]

    return neighbors, nearest_neighbors


def calculate_weight(distance):
    """
    Calculate the weight of a neighbor.

    Weight formula:

        Weight = 1 / Distance

    A closer neighbor gets a higher weight.

    If distance is zero, a very large weight is returned
    so that the exact matching sample gets the highest priority.
    """

    if distance == 0:

        return float("inf")

    else:

        return 1 / distance


def weighted_majority_voting(nearest_neighbors):
    """
    Perform weighted majority voting.

    Instead of giving every neighbor one vote, each neighbor
    receives a weight based on its distance.

    Weight = 1 / Distance

    The class with the highest total weight is selected.

    If multiple classes have the same total weight,
    the class with the smaller average distance is selected.
    """

    weighted_votes = {}

    distance_sum = {}

    distance_count = {}

    for neighbor in nearest_neighbors:

        distance = neighbor[0]

        class_label = neighbor[2]

        weight = calculate_weight(distance)

        if class_label not in weighted_votes:

            weighted_votes[class_label] = 0

            distance_sum[class_label] = 0

            distance_count[class_label] = 0

        weighted_votes[class_label] = (
            weighted_votes[class_label] + weight
        )

        distance_sum[class_label] = (
            distance_sum[class_label] + distance
        )

        distance_count[class_label] = (
            distance_count[class_label] + 1
        )

    maximum_weight = max(weighted_votes.values())

    tied_classes = []

    for class_label in weighted_votes:

        if weighted_votes[class_label] == maximum_weight:

            tied_classes.append(class_label)

    if len(tied_classes) == 1:

        return tied_classes[0], weighted_votes

    best_class = tied_classes[0]

    best_average_distance = (
        distance_sum[best_class] /
        distance_count[best_class]
    )

    for class_label in tied_classes[1:]:

        average_distance = (
            distance_sum[class_label] /
            distance_count[class_label]
        )

        if average_distance < best_average_distance:

            best_class = class_label

            best_average_distance = average_distance

    return best_class, weighted_votes


def weighted_knn_classifier(
        features,
        target,
        test_sample,
        k=5,
        distance_metric="euclidean",
        sorting_method="bubble"):
    """
    Main Weighted kNN function.

    Steps:

        1. Calculate distances
        2. Sort neighbors
        3. Select k nearest neighbors
        4. Calculate weights
        5. Perform weighted voting
        6. Assign predicted class
    """

    all_neighbors, nearest_neighbors = find_nearest_neighbors(
        features,
        target,
        test_sample,
        k,
        distance_metric,
        sorting_method
    )

    predicted_class, weighted_votes = weighted_majority_voting(
        nearest_neighbors
    )

    return (
        all_neighbors,
        nearest_neighbors,
        weighted_votes,
        predicted_class
    )


def main():

    file_name = "features.csv"

    target_column = "person_id"

    identifier_column = "image_name"

    k = 5

    distance_metric = "euclidean"

    sorting_method = "bubble"

    data = pd.read_csv(file_name)

    print("\n================================================")
    print("ORIGINAL DATASET")
    print("================================================")

    print(data)

    data = encode_categorical_columns(
        data,
        target_column
    )

    data = impute_missing_values(data)

    print("\n================================================")
    print("PROCESSED DATASET")
    print("================================================")

    print(data)

    feature_columns = []

    for column in data.columns:

        if (
            column != target_column
            and column != identifier_column
        ):

            feature_columns.append(column)

    features = data[feature_columns]

    target = data[target_column]

    test_index = 0

    test_sample = features.iloc[test_index].tolist()

    actual_class = target.iloc[test_index]

    training_features = features.drop(test_index)

    training_target = target.drop(test_index)

    training_features = training_features.reset_index(drop=True)

    training_target = training_target.reset_index(drop=True)

    (
        all_neighbors,
        nearest_neighbors,
        weighted_votes,
        predicted_class
    ) = weighted_knn_classifier(
        training_features,
        training_target,
        test_sample,
        k,
        distance_metric,
        sorting_method
    )

    print("\n================================================")
    print("DISTANCES")
    print("================================================")

    print("Index\tDistance\tClass")

    for neighbor in all_neighbors:

        print(
            neighbor[1],
            "\t",
            round(neighbor[0], 4),
            "\t",
            neighbor[2]
        )

    print("\n================================================")
    print("SORTED k NEIGHBORS")
    print("================================================")

    print("Rank\tIndex\tDistance\tWeight\tClass")

    for i in range(len(nearest_neighbors)):

        neighbor = nearest_neighbors[i]

        distance = neighbor[0]

        weight = calculate_weight(distance)

        if weight == float("inf"):

            weight_display = "Infinity"

        else:

            weight_display = round(weight, 4)

        print(
            i + 1,
            "\t",
            neighbor[1],
            "\t",
            round(distance, 4),
            "\t",
            weight_display,
            "\t",
            neighbor[2]
        )

    print("\n================================================")
    print("WEIGHTED VOTES")
    print("================================================")

    for class_label in weighted_votes:

        print(
            "Class",
            class_label,
            "-> Total Weight:",
            round(weighted_votes[class_label], 4)
        )

    print("\n================================================")
    print("CLASSIFICATION RESULT")
    print("================================================")

    print("Test Sample Index :", test_index)

    print("Actual Class      :", actual_class)

    print("Predicted Class   :", predicted_class)

    print("\nConfiguration:")

    print("k                 :", k)

    print("Distance Metric   :", distance_metric)

    print("Sorting Algorithm :", sorting_method)

    print("Classification    : Weighted kNN")


if __name__ == "__main__":
    main()