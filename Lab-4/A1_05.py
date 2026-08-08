import pandas as pd
import matplotlib.pyplot as plt

def minkowski_distance(p1, p2, p):
    distance = 0

    for i in range(len(p1)):
        distance += abs(p1[i] - p2[i]) ** p

    return distance ** (1 / p)

data = pd.read_excel("Lab_Session_Data.xlsx", sheet_name="marketing_campaign")

numeric_data = data.select_dtypes(include=["number"])

p1 = numeric_data.iloc[0].values
p2 = numeric_data.iloc[1].values

p_values = []
distances = []

for p in range(1, 11):
    p_values.append(p)
    distances.append(minkowski_distance(p1, p2, p))

plt.xlabel("p value")
plt.ylabel("Minkowski Distance")
plt.title("Minkowski Distance vs p")
plt.grid(True)
plt.show()