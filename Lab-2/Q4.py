# A4

import pandas as pd

def get_data(data):
    return data

def check_datatypes(data):
    return data.dtypes

def check_range(data):
    return data.describe()

def check_missing_values(data):
    return data.isnull().sum()

def check_outliers(data):
    numeric = data.select_dtypes(include="number")
    Q1 = numeric.quantile(0.25)
    Q3 = numeric.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((numeric < (Q1 - 1.5 * IQR)) | (numeric > (Q3 + 1.5 * IQR))).sum()
    return outliers

def cal_mean(data):
    return data.select_dtypes(include="number").mean()

def cal_variance(data):
    return data.select_dtypes(include="number").var()

data = pd.read_excel("Lab_Session_Data.xlsx", sheet_name="thyroid0387_UCI")

data = get_data(data)

print("data types:")
print(check_datatypes(data))

print("\ndata range:")
print(check_range(data))

print("\nmissing values:")
print(check_missing_values(data))

print("\noutliers:")
print(check_outliers(data))

print("\nmean:")
print(cal_mean(data))

print("\nvariance:")
print(cal_variance(data))
