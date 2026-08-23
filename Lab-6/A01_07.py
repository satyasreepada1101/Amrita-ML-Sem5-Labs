import pandas as pd
import math

class KNNClassifier:

    def __init__(self, k=3):
        # Number of nearest neighbours
        self.k = k

        # Training data will be stored here
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

            # Store distance and corresponding class
            distances.append(
                (distance, self.y_train.iloc[i])
            )

        # Sort according to distance
        distances.sort(key=lambda x: x[0])

        # Select k nearest neighbours
        neighbours = distances[:self.k]

        return neighbours

    def vote(self, neighbours):
        votes = {}

        for neighbour in neighbours:

            class_label = neighbour[1]

            if class_label not in votes:

                votes[class_label] = 0

            votes[class_label] = votes[class_label] + 1

        # Find class with maximum votes
        predicted_class = None
        maximum_votes = 0

        for class_label in votes:

            if votes[class_label] > maximum_votes:

                maximum_votes = votes[class_label]

                predicted_class = class_label

        return predicted_class

    def predict(self, X_test):
        predictions = []

        for i in range(len(X_test)):

            test_sample = X_test.iloc[i].tolist()

            # Find nearest neighbours
            neighbours = self.find_neighbours(
                test_sample
            )

            # Perform majority voting
            predicted_class = self.vote(
                neighbours
            )

            predictions.append(predicted_class)

        return predictions

    def score(self, X_test, y_test):

        # Generate predictions
        predictions = self.predict(X_test)

        correct = 0

        for i in range(len(y_test)):

            if predictions[i] == y_test.iloc[i]:

                correct = correct + 1

        accuracy = correct / len(y_test)

        return accuracy


# ============================================================
# MAIN PROGRAM
# ============================================================

# Load the dataset
data = pd.read_csv("features.csv")

# Create feature matrix
# person_id = target
# image_name = identifier, so it is not used as a feature
X = data.drop(columns=["person_id", "image_name"])

# Create target
y = data["person_id"]


# ------------------------------------------------------------
# Split dataset into training and testing data
# ------------------------------------------------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)


# ------------------------------------------------------------
# Create KNN classifier
# ------------------------------------------------------------

knn = KNNClassifier(k=3)


# Train the classifier
knn.fit(X_train, y_train)


# ------------------------------------------------------------
# Predict test samples
# ------------------------------------------------------------

predictions = knn.predict(X_test)


# Display predictions
print("Predicted Labels:")
print(predictions)


# ------------------------------------------------------------
# Calculate accuracy
# ------------------------------------------------------------

accuracy = knn.score(X_test, y_test)

print("\nAccuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100, "%")