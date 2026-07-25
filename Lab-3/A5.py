import pandas as pd
import matplotlib.pyplot as plt


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

# stroing the p values n the corresponding distances 
p_values = []
distances = []

for p in range(1, 11):
    d = minkowski_distance(p1, p2, p)
    p_values.append(p)
    distances.append(d)

print("p Values :", p_values)
print("distances:", distances)

plt.plot(p_values, distances, marker="o")
plt.title("minkowski dist vs p")
plt.xlabel("value of p")
plt.ylabel("distance")
plt.grid(True)
plt.savefig("a5-minkowski_distance.png", dpi=300, bbox_inches="tight")
plt.show()