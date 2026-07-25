import pandas as pd
import numpy as np

def dot_product(v1, v2):
    result = 0
    for i in range(len(v1)):
        result += v1[i] * v2[i]
    return result

def euclidean_norm(vector):
    total = 0
    for value in vector:
        total += value ** 2
    return total ** 0.5

data = pd.read_excel("Lab_Session_Data.xlsx", sheet_name="marketing_campaign")

# only numeric colm data
numeric_data = data.select_dtypes(include=["number"])

# 1st 2-rows as vectors
v1 = numeric_data.iloc[0].values
v2 = numeric_data.iloc[1].values

fun_dot = dot_product(v1, v2)
fun_norm1 = euclidean_norm(v1)
fun_norm2 = euclidean_norm(v2)

numpy_dot = np.dot(v1, v2)
numpy_norm1 = np.linalg.norm(v1)
numpy_norm2 = np.linalg.norm(v2)

print("\ndot product")
print("function value:", fun_dot)
print("numpy:", numpy_dot)

print("\neuclidean norm of vector 1")
print("function value:", fun_norm1)
print("numpy:", numpy_norm1)

print("\neuclidean norm of vector 2")
print("function value:", fun_norm2)
print("numpy:", numpy_norm2)