import pandas as pd
import numpy as np

def find_entropy(data, target_column):
    class_counts = data[target_column].value_counts()
    total_samples = len(data)
    entropy = 0
    for count in class_counts:
        probability = count / total_samples
        entropy -= probability * np.log2(probability)
    return entropy

def equal_width_binning(data, column_name, no_of_bins=5):
    min_value = data[column_name].min()
    max_value = data[column_name].max()
    # creating bin boundaries
    bins = np.linspace(min_value, max_value, no_of_bins + 1)
    # assigning values to bins
    binned_column = pd.cut(data[column_name], bins=bins, labels=False, include_lowest=True)
    return binned_column

def find_information_gain(data, feature_column, target_column):
    total_entropy = find_entropy(data, target_column)
    weighted_entropy = 0
    feature_values = data[feature_column].unique()
    for value in feature_values:
        subset = data[data[feature_column] == value]
        weight = len(subset) / len(data)
        subset_entropy = find_entropy(subset, target_column)
        weighted_entropy += weight * subset_entropy
    information_gain = total_entropy - weighted_entropy
    return information_gain

def find_root_node(data, target_column):
    best_feature = None
    max_gain = -1
    for column in data.columns:
        if column not in [target_column, "image_name"]:
            gain = find_information_gain(data, column, target_column)
            print(column, ":", gain)
            if gain > max_gain:
                max_gain = gain
                best_feature = column
    return best_feature, max_gain


df = pd.read_csv("features.csv")

# binning all feature columns
for column in df.columns:
    if column not in ["person_id", "image_name"]:
        df[column] = equal_width_binning(df, column)
root_feature, gain = find_root_node(df, "person_id")

print("\nroot node =", root_feature)
print("information gain =", gain)