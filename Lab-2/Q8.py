import pandas as pd


def fill_missing_values(data):
    for col in data.columns:

        if pd.api.types.is_numeric_dtype(data[col]):

            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            outlier = ((data[col] < lower) | (data[col] > upper)).any()

            if outlier:
                data[col] = data[col].fillna(data[col].median())
            else:
                data[col] = data[col].fillna(data[col].mean())

        else:
            data[col] = data[col].fillna(data[col].mode()[0])

    return data


data = pd.read_excel("Lab_Session_Data.xlsx", sheet_name="thyroid0387_UCI",na_values=["?"])

filled_data = fill_missing_values(data)

print(filled_data.isnull().sum())