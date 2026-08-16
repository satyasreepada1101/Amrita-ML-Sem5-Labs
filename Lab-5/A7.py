import pandas as pd
import math
from collections import Counter
from sklearn.model_selection import train_test_split

def calculate_distance(row1, row2):
    distance = 0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i]) ** 2
    return math.sqrt(distance)

def insertion_sort(distances):
    for i in range(1, len(distances)):
        key = distances[i]
        j = i - 1
        while j >= 0 and distances[j][1] > key[1]:
            distances[j + 1] = distances[j]
            j -= 1
        distances[j + 1] = key
    return distances

def get_neighbors(X_train, y_train, test_row, k):
    distances = []
    for i in range(len(X_train)):
        dist = calculate_distance(X_train.iloc[i].values, test_row.values)
        distances.append((y_train.iloc[i], dist))
    distances = insertion_sort(distances)
    return distances[:k]

def predict_class(X_train, y_train, test_row, k):
    neighbors = get_neighbors(X_train, y_train, test_row, k)
    labels = []
    for label, dist in neighbors:
        labels.append(label)
    count = Counter(labels)
    return count.most_common(1)[0][0]

class newKNN:

    def __init__(self, k=3):
        self.k = k

    # store training data
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    # predict classes
    def predict(self, X_test):
        predictions = []
        for i in range(len(X_test)):
            pred = predict_class( self.X_train, self.y_train, X_test.iloc[i], self.k )
            predictions.append(pred)
        return predictions

    # find accuracy
    def score(self, X_test, y_test):
        predictions = self.predict(X_test)
        correct = 0
        for i in range(len(y_test)):
            if predictions[i] == y_test.iloc[i]:
                correct += 1
        return correct / len(y_test)

df = pd.read_csv("features.csv")
df = df[df["person_id"].isin(["A", "B"])]

X = df.drop( ["person_id", "image_name"], axis=1 )
y = df["person_id"]

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=42 )

model = newKNN(3)
model.fit(X_train, y_train)

accuracy = model.score( X_test, y_test )
print("\naccuracy:", accuracy)