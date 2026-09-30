import pandas as pd
import numpy as np

def summation_unit(inputs, weights, bias=0):
    total = bias
    for x, w in zip(inputs, weights):
        total += x * w
    return total

def step_activation(x):
    return 1 if x >= 0 else 0

def bipolar_step_activation(x):
    if x > 0:
        return 1
    elif x == 0:
        return 0
    else:
        return -1

def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))

def tanh_activation(x):
    return np.tanh(x)

def relu_activation(x):
    return max(0, x)

def leaky_relu_activation(x, alpha=0.01):
    return x if x > 0 else alpha * x

def comparator_unit(target, predicted):
    return target - predicted

df = pd.read_csv("features.csv")
sample = df.iloc[0]

inputs = [
    sample["top_left_R_mean"],
    sample["top_left_G_mean"],
    sample["top_left_B_mean"]
]

weights = [0.2, -0.1, 0.3]
bias = 0.5

net_input = summation_unit(inputs, weights, bias)

print("Net Input =", net_input)
print("Step =", step_activation(net_input))
print("Bipolar Step =", bipolar_step_activation(net_input))
print("Sigmoid =", sigmoid_activation(net_input))
print("Tanh =", tanh_activation(net_input))
print("ReLU =", relu_activation(net_input))
print("Leaky ReLU =", leaky_relu_activation(net_input))

target = 1
predicted = step_activation(net_input)

error = comparator_unit(target, predicted)

print("Error =", error)