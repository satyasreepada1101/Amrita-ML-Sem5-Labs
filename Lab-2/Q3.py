# A3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time


def calculate_mean(price):
    return np.mean(price)


def calculate_variance(price):
    return np.var(price)


def my_mean(price):
    return sum(price) / len(price)


def my_variance(price):
    mean = my_mean(price)
    total = 0
    for i in price:
        total += (i - mean) ** 2
    return total / len(price)


def average_time(func, data):
    times = []

    for i in range(10):
        start = time.time()
        func(data)
        end = time.time()
        times.append(end - start)

    return sum(times) / len(times)


data = pd.read_excel("Lab_Session_Data.xlsx", sheet_name="IRCTC Stock Price")

price = data["Price"]

print("mean using NumPy:", calculate_mean(price))
print("variance using NumPy:", calculate_variance(price))

print("\nmean using function:", my_mean(price))
print("variance using function:", my_variance(price))

print("\naverage time (numpy's mean):", average_time(calculate_mean, price))
print("average time (my mean):", average_time(my_mean, price))

print("\naverage time (numpy's variance):", average_time(calculate_variance, price))
print("average time (my variance):", average_time(my_variance, price))

wednesday = data[data["Day"] == "Wed"]
print("\nwed mean:", calculate_mean(wednesday["Price"]))

april = data[data["Month"] == "Apr"]
print("april mean:", calculate_mean(april["Price"]))

loss = data["Chg%"].apply(lambda x: x < 0)
loss_probability = loss.sum() / len(data)
print("\nprobability of loss:", loss_probability)

wednesday_profit = wednesday["Chg%"].apply(lambda x: x > 0)
profit_wednesday = wednesday_profit.sum() / len(data)
print("probability of profit on wed:", profit_wednesday)

conditional_probability = wednesday_profit.sum() / len(wednesday)
print("conditional probability of profit given wed:", conditional_probability)

plt.scatter(data["Day"], data["Chg%"])
plt.xlabel("day")
plt.ylabel("Chg%")
plt.title("Chg% vs day")
plt.show()
