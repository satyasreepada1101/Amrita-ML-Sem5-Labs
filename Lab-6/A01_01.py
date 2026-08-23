import pandas as pd
import math


def label_encode_column(data, column_name):
    unique_values = data[column_name].dropna().unique()
    encoding = {}

    for i in range(len(unique_values)):
        encoding[unique_values[i]] = i

    data[column_name] = data[column_name].map(encoding)
    return data


def encode_categorical_columns(data, target_column):
    data = data.copy()

    for column in data.columns:
        if data[column].dtype == "object":
            data = label_encode_column(data, column)

    return data


def check_outliers(column):
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
    skewness = column.skew()

    if -0.5 <= skewness <= 0.5:
        return True
    else:
        return False


def impute_missing_values(data):
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
    total = 0

    for i in range(len(sample1)):
        difference = sample1[i] - sample2[i]
        total = total + (difference * difference)

    return math.sqrt(total)


def manhattan_distance(sample1, sample2):
    total = 0

    for i in range(len(sample1)):
        difference = abs(sample1[i] - sample2[i])
        total = total + difference

    return total


def minkowski_distance(sample1, sample2, p=3):
    total = 0

    for i in range(len(sample1)):
        difference = abs(sample1[i] - sample2[i])
        total = total + (difference ** p)

    return total ** (1 / p)


def calculate_distance(sample1, sample2, metric="euclidean"):
    if metric == "euclidean":
        return euclidean_distance(sample1, sample2)

    elif metric == "manhattan":
        return manhattan_distance(sample1, sample2)

    elif metric == "minkowski":
        return minkowski_distance(sample1, sample2)

    else:
        print("Invalid distance metric.")
        print("Use euclidean, manhattan or minkowski.")
        return None


def bubble_sort(neighbors):
    n = len(neighbors)

    for i in range(n):
        for j in range(0, n - i - 1):

            if (neighbors[j][0] > neighbors[j + 1][0] or
                    (neighbors[j][0] == neighbors[j + 1][0] and
                     neighbors[j][1] > neighbors[j + 1][1])):

                temp = neighbors[j]
                neighbors[j] = neighbors[j + 1]
                neighbors[j + 1] = temp

    return neighbors


def selection_sort(neighbors):
    n = len(neighbors)

    for i in range(n):
        smallest = i

        for j in range(i + 1, n):

            if (neighbors[j][0] < neighbors[smallest][0] or
                    (neighbors[j][0] == neighbors[smallest][0] and
                     neighbors[j][1] < neighbors[smallest][1])):

                smallest = j

        temp = neighbors[i]
        neighbors[i] = neighbors[smallest]
        neighbors[smallest] = temp

    return neighbors


def insertion_sort(neighbors):
    for i in range(1, len(neighbors)):

        current = neighbors[i]
        j = i - 1

        while j >= 0:

            should_move = (
                neighbors[j][0] > current[0] or
                (
                    neighbors[j][0] == current[0] and
                    neighbors[j][1] > current[1]
                )
            )

            if should_move:
                neighbors[j + 1] = neighbors[j]
                j = j - 1
            else:
                break

        neighbors[j + 1] = current

    return neighbors


def sort_neighbors(neighbors, sorting_method="bubble"):
    if sorting_method == "bubble":
        return bubble_sort(neighbors)

    elif sorting_method == "selection":
        return selection_sort(neighbors)

    elif sorting_method == "insertion":
        return insertion_sort(neighbors)

    else:
        print("Invalid sorting method.")
        print("Use bubble, selection or insertion.")
        return neighbors


def find_nearest_neighbors(
        features,
        target,
        test_sample,
        k,
        distance_metric="euclidean",
        sorting_method="bubble"):

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


def majority_voting(nearest_neighbors):
    vote_count = {}
    distance_sum = {}
    distance_count = {}

    for neighbor in nearest_neighbors:

        distance = neighbor[0]
        class_label = neighbor[2]

        if class_label not in vote_count:
            vote_count[class_label] = 0
            distance_sum[class_label] = 0
            distance_count[class_label] = 0

        vote_count[class_label] = vote_count[class_label] + 1
        distance_sum[class_label] = (
            distance_sum[class_label] + distance
        )
        distance_count[class_label] = (
            distance_count[class_label] + 1
        )

    maximum_votes = max(vote_count.values())

    tied_classes = []

    for class_label in vote_count:
        if vote_count[class_label] == maximum_votes:
            tied_classes.append(class_label)

    if len(tied_classes) == 1:
        return tied_classes[0]

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

    return best_class


def knn_classifier(
        features,
        target,
        test_sample,
        k=5,
        distance_metric="euclidean",
        sorting_method="bubble"):

    all_neighbors, nearest_neighbors = find_nearest_neighbors(
        features,
        target,
        test_sample,
        k,
        distance_metric,
        sorting_method
    )

    predicted_class = majority_voting(
        nearest_neighbors
    )

    return all_neighbors, nearest_neighbors, predicted_class


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
        if column != target_column and column != identifier_column:
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

    all_neighbors, nearest_neighbors, predicted_class = (
        knn_classifier(
            training_features,
            training_target,
            test_sample,
            k,
            distance_metric,
            sorting_method
        )
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
    print("SORTED NEIGHBORS")
    print("================================================")

    print("Rank\tIndex\tDistance\tClass")

    for i in range(len(nearest_neighbors)):

        neighbor = nearest_neighbors[i]

        print(
            i + 1,
            "\t",
            neighbor[1],
            "\t",
            round(neighbor[0], 4),
            "\t",
            neighbor[2]
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


if __name__ == "__main__":
    main()
