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


# for a categorical column, has binary values 0 or 1, with 1 colm is "1" for each row
def one_hot_encode(column):
    unique_values = column.unique()
    encoded_data = {}
    # an empty list for each category name
    for value in unique_values:
        encoded_data[value] = []
    # in each row, add 1 to the matching category column and 0 to the others
    for item in column:
        for value in unique_values:
            if item == value:
                encoded_data[value].append(1)
            else:
                encoded_data[value].append(0)
    return pd.DataFrame(encoded_data)

data = pd.read_excel("Lab_Session_Data.xlsx", sheet_name="marketing_campaign")

column = data["Education"]

label_encoded = label_encode(column)
print("label encoded:")
print(label_encoded)

print()

one_hot_encoded = one_hot_encode(column)
print("one hot encoded:")
print(one_hot_encoded)