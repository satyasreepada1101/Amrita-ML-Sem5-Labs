import pandas as pd

# convert categorical values into integer labels, each unique category is assigned a different number such as 0, 1, 2...etc
def label_encode(column):
    unique_values = column.unique()
    encoding = {}
    # assign a num to each unique category
    for i in range(len(unique_values)):
        encoding[unique_values[i]] = i
    encoded_column = []
    # replace every category value with its assigned num
    for value in column:
        encoded_column.append(encoding[value])
    return encoded_column

# for a categorical colm, has binary values 0 or 1, with 1 colm is "1" for each row
def one_hot_encode(column):
    unique_values = column.unique()
    encoded_data = {}
    # an empty list for each category name
    for value in unique_values:
        encoded_data[value] = []
    # in each row, add 1 to the matching category colm and 0 to the others
    for item in column:
        for value in unique_values:
            if item == value:
                encoded_data[value].append(1)
            else:
                encoded_data[value].append(0)
    return pd.DataFrame(encoded_data)

data = pd.read_excel("Lab_Session_Data.xlsx", sheet_name="marketing_campaign")

print("original dataset shape:", data.shape)

# to find all categorical colms
categorical_columns = data.select_dtypes(include=["object"]).columns

# label encoding to replace each categorical colm with numeric labels
label_data = data.copy()

for column in categorical_columns:
    label_data[column] = label_encode(label_data[column])

print("after label:", label_data.shape)

# one hot, making new colm for each category
one_hot_data = data.copy()

for column in categorical_columns:
    encoded = one_hot_encode(one_hot_data[column])

    one_hot_data = one_hot_data.drop(column, axis=1)
    one_hot_data = pd.concat([one_hot_data, encoded], axis=1)

print("after one hot:", one_hot_data.shape)