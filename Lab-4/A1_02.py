import pandas as pd

# Read the Excel file
data = pd.read_excel("Lab_Session_Data.xlsx", sheet_name="marketing_campaign")

# Select a sample categorical feature
categorical_columns = data.select_dtypes(include=["object"]).columns

feature_name = categorical_columns[0]
feature = data[feature_name]


# Function for Label Encoding
def label_encode(column_data):
    unique_values = []

    for value in column_data:
        if value not in unique_values:
            unique_values.append(value)

    mapping = {}

    for i in range(len(unique_values)):
        mapping[unique_values[i]] = i

    encoded = []

    for value in column_data:
        encoded.append(mapping[value])

    return encoded, mapping


# Function for One-Hot Encoding
def one_hot_encode(column_data):
    unique_values = []

    for value in column_data:
        if value not in unique_values:
            unique_values.append(value)

    encoded = []

    for value in column_data:
        row = []

        for category in unique_values:
            if value == category:
                row.append(1)
            else:
                row.append(0)

        encoded.append(row)

    return encoded, unique_values


# Perform Label Encoding
label_encoded, label_mapping = label_encode(feature)

# Perform One-Hot Encoding
one_hot_encoded, categories = one_hot_encode(feature)

# Display results
print("Selected Feature:", feature_name)

print("\nLabel Encoding Mapping:")
print(label_mapping)

print("\nLabel Encoded Values:")
print(label_encoded)

print("\nOne-Hot Encoding Categories:")
print(categories)

print("\nOne-Hot Encoded Values:")
for row in one_hot_encoded:
    print(row)
