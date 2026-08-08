import pandas as pd
from scipy.spatial.distance import minkowski

def my_distance(p1, p2, p):
    distance = 0

    for i in range(len(p1)):
        distance += abs(p1[i] - p2[i]) ** p

    return distance ** (1 / p)


def compare_distance(p1, p2, p):
    my_result = my_distance(p1, p2, p)
    scipy_result = minkowski(p1, p2, p)

    print("My Minkowski Distance =", my_result)
    print("Scipy Minkowski Distance =", scipy_result)

    if abs(my_result - scipy_result) < 1e-6:
        print("Both values match.")
    else:
        print("Values do not match.")


data = pd.read_excel("Lab_Session_Data.xlsx", sheet_name="marketing_campaign")

numeric_data = data.select_dtypes(include=["number"])

p1 = numeric_data.iloc[0].values
p2 = numeric_data.iloc[1].values

compare_distance(p1, p2, 3)