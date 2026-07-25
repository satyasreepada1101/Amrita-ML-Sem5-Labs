import pandas as pd
import numpy as np
import math

def find_mean(data):
    total = 0
    for value in data:
        total += value
    return total / len(data)

def find_variance(data):
    mean = find_mean(data)
    total = 0
    for value in data:
        total += (value - mean) ** 2
    return total / len(data)

def find_std(data):
    variance = find_variance(data)
    return math.sqrt(variance)

data = pd.read_excel("Lab_Session_Data.xlsx", sheet_name="marketing_campaign")

numeric_data = data.select_dtypes(include=["number"])

for column in numeric_data.columns:
    values = numeric_data[column].dropna().tolist()

    fun_mean = find_mean(values)
    fun_std = find_std(values)

    numpy_mean = np.mean(values)
    numpy_std = np.std(values)

    print("feature:", column)
    print("function mean:", fun_mean)
    print("numpy mean:", numpy_mean)
    print("function std:", fun_std)
    print("numpy std:", numpy_std)
    print()