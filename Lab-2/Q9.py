import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def fill_missing_values(data):
    filled_data = data.copy()

    for col in filled_data.columns:

        if pd.api.types.is_numeric_dtype(filled_data[col]):

            Q1 = filled_data[col].quantile(0.25)
            Q3 = filled_data[col].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            has_outlier = ((filled_data[col] < lower) | (filled_data[col] > upper)).any()

            if has_outlier:
                filled_data[col] = filled_data[col].fillna(filled_data[col].median())
            else:
                filled_data[col] = filled_data[col].fillna(filled_data[col].mean())

        else:
            filled_data[col] = filled_data[col].fillna(filled_data[col].mode()[0])

    return filled_data


def normalize_data(data):
    normalized_data = data.copy()

    numeric_columns = normalized_data.select_dtypes(include=np.number).columns

    if "Record ID" in numeric_columns:
        numeric_columns = numeric_columns.drop("Record ID")

    scaler = MinMaxScaler()
    normalized_data[numeric_columns] = scaler.fit_transform(normalized_data[numeric_columns])

    return normalized_data


data = pd.read_excel(
    "Lab_Session_Data.xlsx",
    sheet_name="thyroid0387_UCI",
    na_values=["?"]
)

filled_data = fill_missing_values(data)

normalized_data = normalize_data(filled_data)

print("data after normalization:")
print(normalized_data.head())