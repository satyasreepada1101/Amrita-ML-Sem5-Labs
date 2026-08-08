import pandas as pd

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


# Calculate statistics for every numeric column
results = []

for column in numeric_data.columns:
    values = numeric_data[column].dropna().values

    mean = calc_mean(values)
    variance = calc_variance(values)
    std = calc_std(values)

    results.append([column, mean, variance, std])


# Display results in tabular format
result_df = pd.DataFrame(
    results,
    columns=["Column", "Mean", "Variance", "Standard Deviation"]
)

print(result_df)