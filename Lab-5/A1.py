# knn algo

import pandas as pd
import math
from collections import Counter

def encode_data(df):
    return df

def impute_missing_values(df):
    return df.fillna(df.mean(numeric_only=True))

def calculate_distance(row1, row2):
    distance = 0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i]) ** 2
    return math.sqrt(distance)

def bubble_sort(distances):
    n = len(distances)
    for i in range(n):
        for j in range(0, n - i - 1):
            if distances[j][1] > distances[j + 1][1]:
                distances[j], distances[j + 1] = distances[j + 1], distances[j]
    return distances

def selection_sort(distances):
    n = len(distances)
    for i in range(n):
        min_index = i
        for j in range(i + 1, n):
            if distances[j][1] < distances[min_index][1]:
                min_index = j
        distances[i], distances[min_index] = distances[min_index], distances[i]
    return distances

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
        dist = calculate_distance(X_train.iloc[i].values,test_row.values)
        distances.append((y_train.iloc[i], dist))
    distances = insertion_sort(distances)
    return distances[:k]

def predict_class(X_train, y_train, test_row, k):
    neighbors = get_neighbors(X_train,y_train,test_row,k)
    labels = []
    for label, dist in neighbors:
        labels.append(label)
    count = Counter(labels)
    max_votes = max(count.values())
    winners = []
    for label, votes in count.items():
        if votes == max_votes:
            winners.append(label)
    winners.sort()
    return winners[0]

# main
df = pd.read_csv("features.csv")
df = encode_data(df)
df = impute_missing_values(df)
df = df[df["person_id"].isin(["A", "B"])]

X = df.drop(["person_id", "image_name"], axis=1)
y = df["person_id"]

test_row = X.iloc[0]
prediction = predict_class(X,y,test_row,3)

print("\nPredicted Class:", prediction)