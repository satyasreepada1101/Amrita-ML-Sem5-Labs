#A1

import pandas as pd
import numpy as np

def get_matrices(data):
    X = data[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
    y = data["Payment (Rs)"].values
    return X, y

def calculate_rank(X):
    return np.linalg.matrix_rank(X)

def calculate_cost(X, y):
    X_pinv = np.linalg.pinv(X)
    cost = X_pinv.dot(y)
    return cost

data = pd.read_excel("Lab_Session_Data.xlsx", sheet_name="Purchase data")

X, y = get_matrices(data)

print("feature matrix X:")
print(X)

print("\noutput vector y:")
print(y)

rank = calculate_rank(X)
print("rank of feature matrix:", rank)

cost = calculate_cost(X, y)

print("\ncandy cost:", cost[0])
print("mango cost:", cost[1])
print("milk cost:", cost[2])
