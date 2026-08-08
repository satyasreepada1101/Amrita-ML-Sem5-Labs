import pandas as pd
import numpy as np

# Read the Excel file
data = pd.read_excel("Lab_Session_Data.xlsx", sheet_name="marketing_campaign")

# Select only numeric columns
numeric_data = data.select_dtypes(include=["number"])


# Function to calculate mean
def calc_mean(data):
    total = 0

    for value in data:
        total += value

    return total / len(data)


# Function to calculate variance
def calc_variance(data):
    mean = calc_mean(data)
    variance = 0

    for value in data:
        variance += (value - mean) ** 2

    return variance / len(data)


# Function to calculate standard deviation
def calc_std(data):
    variance = calc_variance(data)
    return variance ** 0.5


# Function to compare statistics
def compare_stats(data):
    # NumPy calculations
    numpy_mean = np.mean(data, axis=0)
    numpy_std = np.std(data, axis=0)

    # My calculations
    my_mean = []
    my_std = []

    for column in data.columns:
        values = data[column].dropna().values
        my_mean.append(calc_mean(values))
        my_std.append(calc_std(values))

    # Display comparison
    print("{:<25} {:>15} {:>15} {:>15} {:>15}".format(
        "Column", "My Mean", "NumPy Mean", "My Std", "NumPy Std"))

    print("-" * 90)

    for i, column in enumerate(data.columns):
        print("{:<25} {:>15.4f} {:>15.4f} {:>15.4f} {:>15.4f}".format(
            column,
            my_mean[i],
            numpy_mean[i],
            my_std[i],
            numpy_std[i]
        ))

    print("\nComparison Result:")
    for i, column in enumerate(data.columns):
        if abs(my_mean[i] - numpy_mean[i]) < 1e-6 and abs(my_std[i] - numpy_std[i]) < 1e-6:
            print(column, ": Values Match")
        else:
            print(column, ": Values Do Not Match")


# Call the function
compare_stats(numeric_data)