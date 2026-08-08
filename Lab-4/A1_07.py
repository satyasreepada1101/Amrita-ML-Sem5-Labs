import pandas as pd
import numpy as np

# Read the Excel file
data = pd.read_excel("Lab_Session_Data.xlsx", sheet_name="marketing_campaign")

# Select only numeric columns
numeric_data = data.select_dtypes(include=["number"])

# Take the first two rows as vectors
a = numeric_data.iloc[0].values
b = numeric_data.iloc[1].values


# Function to calculate dot product
def my_dot(a, b):
    dot = 0

    for i in range(len(a)):
        dot += a[i] * b[i]

    return dot


# Function to calculate Euclidean norm
def my_norm(v):
    norm = 0

    for value in v:
        norm += value ** 2

    return norm ** 0.5


# My results
my_dot_result = my_dot(a, b)
my_norm_a = my_norm(a)
my_norm_b = my_norm(b)

# NumPy results
numpy_dot_result = np.dot(a, b)
numpy_norm_a = np.linalg.norm(a)
numpy_norm_b = np.linalg.norm(b)

# Print comparison
print("Dot Product Comparison")
print("My Dot Product      :", my_dot_result)
print("NumPy Dot Product   :", numpy_dot_result)

print("\nNorm Comparison")
print("My Norm (Vector A)  :", my_norm_a)
print("NumPy Norm (Vector A):", numpy_norm_a)

print("My Norm (Vector B)  :", my_norm_b)
print("NumPy Norm (Vector B):", numpy_norm_b)