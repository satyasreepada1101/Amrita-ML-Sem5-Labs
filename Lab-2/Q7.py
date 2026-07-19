import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


def get_binary_data(data):
    binary_columns = []

    for col in data.columns:
        if len(data[col].dropna().unique()) == 2:
            binary_columns.append(col)

    binary_data = data[binary_columns].copy()

    for col in binary_columns:
        values = binary_data[col].dropna().unique()

        if not set(values).issubset({0, 1}):
            binary_data[col] = binary_data[col].replace({
                values[0]: 0,
                values[1]: 1
            })

    return binary_data


def encode_data(data):
    encoded = data.copy()

    for col in encoded.columns:
        if not pd.api.types.is_numeric_dtype(encoded[col]):
            encoder = LabelEncoder()
            encoded[col] = encoder.fit_transform(encoded[col].astype(str))

    return encoded


def jaccard(v1, v2):
    f11 = f10 = f01 = 0

    for i in range(len(v1)):
        if v1[i] == 1 and v2[i] == 1:
            f11 += 1
        elif v1[i] == 1 and v2[i] == 0:
            f10 += 1
        elif v1[i] == 0 and v2[i] == 1:
            f01 += 1

    if (f11 + f10 + f01) == 0:
        return 0

    return f11 / (f11 + f10 + f01)


def smc(v1, v2):
    f00 = f01 = f10 = f11 = 0

    for i in range(len(v1)):
        if v1[i] == 0 and v2[i] == 0:
            f00 += 1
        elif v1[i] == 0 and v2[i] == 1:
            f01 += 1
        elif v1[i] == 1 and v2[i] == 0:
            f10 += 1
        else:
            f11 += 1

    return (f11 + f00) / (f00 + f01 + f10 + f11)


def cosine(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


data = pd.read_excel("Lab_Session_Data.xlsx", sheet_name="thyroid0387_UCI")

binary_data = get_binary_data(data)
encoded_data = encode_data(data)

binary_data = binary_data.iloc[:20].values
encoded_data = encoded_data.iloc[:20].values

jc_matrix = np.zeros((20, 20))
smc_matrix = np.zeros((20, 20))
cos_matrix = np.zeros((20, 20))

for i in range(20):
    for j in range(20):
        jc_matrix[i][j] = jaccard(binary_data[i], binary_data[j])
        smc_matrix[i][j] = smc(binary_data[i], binary_data[j])
        cos_matrix[i][j] = cosine(encoded_data[i], encoded_data[j])

plt.figure(figsize=(8,6))
sns.heatmap(jc_matrix)
plt.title("jaccard coeff")
plt.savefig("q7-jaccard_coeff.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(smc_matrix)
plt.title("simple matching coeff")
plt.savefig("q7-smc_coeff.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(cos_matrix)
plt.title("cosine similarity")
plt.savefig("q7-cosine_similarity.png", dpi=300, bbox_inches="tight")
plt.show()