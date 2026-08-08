import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the Excel file
data = pd.read_excel("Lab_Session_Data.xlsx", sheet_name="marketing_campaign")

# Select only numeric columns
numeric_data = data.select_dtypes(include=["number"])

# Choose any one numeric feature
feature_name = numeric_data.columns[0]
feature = numeric_data[feature_name].dropna().values


# Function to calculate mean and variance
def feature_stats(data):
    total = 0

    for value in data:
        total += value

    mean = total / len(data)

    variance = 0

    for value in data:
        variance += (value - mean) ** 2

    variance = variance / len(data)

    return mean, variance


# Calculate statistics
mean, variance = feature_stats(feature)

# Generate histogram information
hist, bins = np.histogram(feature)

# Display results
print("Selected Feature :", feature_name)
print("\nHistogram Counts :", hist)
print("Histogram Bins   :", bins)
print("\nMean             :", mean)
print("Variance         :", variance)

# Plot histogram
plt.hist(feature, bins=10, edgecolor="black")
plt.title("Histogram of " + feature_name)
plt.xlabel(feature_name)
plt.ylabel("Frequency")
plt.grid(True)
plt.show()