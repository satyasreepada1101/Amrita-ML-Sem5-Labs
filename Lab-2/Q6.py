import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def encode_data(data):
    encoded_data = data.copy()

    for col in encoded_data.columns:
        if not pd.api.types.is_numeric_dtype(encoded_data[col]):
            encoder = LabelEncoder()
            encoded_data[col] = encoder.fit_transform(encoded_data[col].astype(str))

    return encoded_data


def get_vectors(data):
    v1 = data.iloc[0].values
    v2 = data.iloc[1].values
    return v1, v2


def calculate_cosine(v1, v2):
    dot_product = np.dot(v1, v2)
    magnitude1 = np.linalg.norm(v1)
    magnitude2 = np.linalg.norm(v2)

    cosine = dot_product / (magnitude1 * magnitude2)
    return cosine


data = pd.read_excel("Lab_Session_Data.xlsx", sheet_name="thyroid0387_UCI")

encoded_data = encode_data(data)

v1, v2 = get_vectors(encoded_data)

cosine = calculate_cosine(v1, v2)

print("Vector 1:")
print(v1)
print("Vector 2:")
print(v2)

print("\nCosine Similarity:", cosine)