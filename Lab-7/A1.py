import pandas as pd
import numpy as np

def find_entropy(data, target_column):
    # counting the no of samples for each class
    class_counts = data[target_column].value_counts()
    # total sample length
    total_samples = len(data)
    entropy = 0
    # traverse through each class
    for count in class_counts:
        probability = count / total_samples
        # entropy 
        entropy -= probability * np.log2(probability)
    return entropy

df = pd.read_csv("features.csv")
entropy_value = find_entropy(df, "person_id")
print("\nentropy =", entropy_value)