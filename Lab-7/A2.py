import pandas as pd

def find_gini(data, target_column):
    class_counts = data[target_column].value_counts()
    total_samples = len(data)
    gini = 1
    for count in class_counts:
        probability = count / total_samples
        # calculate gini
        gini -= probability ** 2
    return gini

df = pd.read_csv("features.csv")

gini_value = find_gini(df, "person_id")

print("\ngini index =", gini_value)