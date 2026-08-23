import pandas as pd
import math
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from A01_07 import KNNClassifier

class WeightedKNNClassifier:

    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def calculate_distance(self, sample1, sample2):

        total = 0

        for i in range(len(sample1)):

            difference = sample1[i] - sample2[i]

            total = total + difference ** 2

        return math.sqrt(total)

    def find_neighbours(self, test_sample):

        distances = []

        for i in range(len(self.X_train)):

            train_sample = self.X_train.iloc[i].tolist()

            distance = self.calculate_distance(
                test_sample,
                train_sample
            )

            class_label = self.y_train.iloc[i]

            distances.append(
                (distance, class_label)
            )

        # Sort neighbours according to distance
        distances.sort(key=lambda x: x[0])

        # Select k nearest neighbours
        return distances[:self.k]

    def predict_one(self, test_sample):

        neighbours = self.find_neighbours(test_sample)

        weighted_votes = {}

        for distance, class_label in neighbours:

            # Inverse distance weighting
            if distance == 0:
                weight = float("inf")
            else:
                weight = 1 / distance

            if class_label not in weighted_votes:

                weighted_votes[class_label] = 0

            weighted_votes[class_label] += weight

        # Find class with highest total weight
        predicted_class = None
        highest_weight = -1

        for class_label in weighted_votes:

            if weighted_votes[class_label] > highest_weight:

                highest_weight = weighted_votes[class_label]

                predicted_class = class_label

        return predicted_class

    def predict(self, X_test):

        predictions = []

        for i in range(len(X_test)):

            test_sample = X_test.iloc[i].tolist()

            predicted_class = self.predict_one(
                test_sample
            )

            predictions.append(predicted_class)

        return predictions

    def score(self, X_test, y_test):

        predictions = self.predict(X_test)

        correct = 0

        for i in range(len(y_test)):

            if predictions[i] == y_test.iloc[i]:

                correct += 1

        accuracy = correct / len(y_test)

        return accuracy

data = pd.read_csv("features.csv")

X = data.drop(
    columns=["person_id", "image_name"]
)

# Create target
y = data["person_id"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

k_values = [1, 3, 5, 7, 9]

normal_accuracy = []
weighted_accuracy = []

for k in k_values:
    
    normal_knn = KNNClassifier(k=k)

    normal_knn.fit(
        X_train,
        y_train
    )

    normal_score = normal_knn.score(
        X_test,
        y_test
    )

    normal_accuracy.append(normal_score)

    weighted_knn = WeightedKNNClassifier(k=k)

    weighted_knn.fit(
        X_train,
        y_train
    )

    weighted_score = weighted_knn.score(
        X_test,
        y_test
    )

    weighted_accuracy.append(weighted_score)

print("\nAccuracy Comparison")
print("----------------------------------------")

print("k\tNormal kNN\tWeighted kNN")
print("----------------------------------------")

for i in range(len(k_values)):

    print(
        k_values[i],
        "\t",
        round(normal_accuracy[i], 4),
        "\t\t",
        round(weighted_accuracy[i], 4)
    )

plt.plot(
    k_values,
    normal_accuracy,
    marker="o",
    label="Normal kNN"
)

plt.plot(
    k_values,
    weighted_accuracy,
    marker="o",
    label="Weighted kNN"
)

plt.xlabel("Value of k")
plt.ylabel("Accuracy")

plt.title(
    "Normal kNN vs Weighted kNN"
)

plt.xticks(k_values)
plt.legend()
plt.grid()
plt.show()