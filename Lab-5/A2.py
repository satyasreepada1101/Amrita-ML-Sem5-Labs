# weighted knn

import pandas as pd
import math

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
        dist = calculate_distance(X_train.iloc[i].values,test_row.values)
        distances.append((y_train.iloc[i], dist))
    distances = insertion_sort(distances)
    return distances[:k]

def weighted_knn(X_train, y_train, test_row, k):
    neighbors = get_neighbors(X_train,y_train,test_row,k)
    class_weights = {}

    for label, distance in neighbors:
        # to calculate weight
        if distance == 0: 
            weight = 999999
        else:
            weight = 1 / distance

        # to add weight for each class
        if label not in class_weights:
            class_weights[label] = 0
        class_weights[label] += weight

    
    # class with highest weight is selected
    predicted_class = max(class_weights,key=class_weights.get) 
    return predicted_class

# main
df = pd.read_csv("features.csv")
df = df[df["person_id"].isin(["A", "B"])]

X = df.drop(["person_id", "image_name"],axis=1)
y = df["person_id"]

test_row = X.iloc[0]
prediction = weighted_knn(X,y,test_row,3)

print("Predicted Class:", prediction)