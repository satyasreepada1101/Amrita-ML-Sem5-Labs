import pandas as pd

# Read the Excel file
data = pd.read_excel("Lab_Session_Data.xlsx", sheet_name="marketing_campaign")


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

    return encoded


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


# Number of features before encoding
features_before = data.shape[1]

# Create encoded dataset
encoded_data = pd.DataFrame()

for column in data.columns:

    if data[column].dtype == "object":

        encoded_values, categories = one_hot_encode(data[column])

        for i in range(len(categories)):
            new_column = []

            for row in encoded_values:
                new_column.append(row[i])

            encoded_data[column + "_" + str(categories[i])] = new_column

    else:
        encoded_data[column] = data[column]

# Number of features after encoding
features_after = encoded_data.shape[1]

# Display results
print("Number of Features Before Encoding :", features_before)
print("Number of Features After Encoding  :", features_after)
print("Encoded Dataset Shape              :", encoded_data.shape)

print("\nEncoded Dataset:")
print(encoded_data.head())