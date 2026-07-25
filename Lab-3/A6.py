import pandas as pd
from scipy.spatial.distance import minkowski

def minkowski_distance(p1, p2, p):
    distance = 0
    for i in range(len(p1)):
        distance += abs(p1[i] - p2[i]) ** p
    return float(distance ** (1 / p))

data = pd.read_excel("Lab_Session_Data.xlsx", sheet_name="marketing_campaign")

numeric_data = data.select_dtypes(include=["number"])

# taking 1st 2-rows as 2 pointers to compare
p1 = numeric_data.iloc[0].values
p2 = numeric_data.iloc[1].values
p = 2

my_dist = minkowski_distance(p1, p2, p)
package_dist = minkowski(p1, p2, p)

print("function value:", my_dist)
print("in built way value:", package_dist)