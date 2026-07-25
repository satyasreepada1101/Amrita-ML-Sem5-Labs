import pandas as pd
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

def dataset_statistics(dataset):
    mean_list = []
    variance_list = []
    std_list = []

    for column in dataset.columns:
        values = dataset[column].dropna().tolist()

        mean_list.append(find_mean(values))
        variance_list.append(find_variance(values))
        std_list.append(find_std(values))

    return mean_list, variance_list, std_list

data = pd.read_excel("Lab_Session_Data.xlsx", sheet_name="marketing_campaign")

numeric_data = data.select_dtypes(include=["number"])

mean_values, variance_values, std_values = dataset_statistics(numeric_data)

for i in range(len(numeric_data.columns)):
    print("feature:", numeric_data.columns[i])
    print("mean:", mean_values[i])
    print("variance:", variance_values[i])
    print("standard deviation:", std_values[i])
    print()