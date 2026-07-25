import pandas as pd
import matplotlib.pyplot as plt

def find_mean(data):
    total = 0
    for value in data:
        total += value
    return total / len(data)

def find_variance(data):
    mean = find_mean(data)
    total = 0
    for value in data:
        total += (value - mean) ** 2
    return total / len(data)

data = pd.read_excel("Lab_Session_Data.xlsx", sheet_name="marketing_campaign")

# taking "income" colm n removing empty values
feature = data["Income"].dropna()

mean = find_mean(feature)
variance = find_variance(feature)

print("for income")
print("mean:", mean)
print("variance:", variance)

plt.hist(feature, bins=10, edgecolor="black")
plt.xlabel("income")
plt.ylabel("frequency")
plt.grid(True)
plt.show()