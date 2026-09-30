import numpy as np
from sklearn.preprocessing import MinMaxScaler

X = np.array([
    [20, 6, 2, 386],
    [16, 3, 6, 289],
    [27, 6, 2, 393],
    [19, 1, 2, 110],
    [24, 4, 2, 280],
    [22, 1, 5, 167],
    [15, 4, 2, 271],
    [18, 4, 2, 274],
    [21, 1, 4, 148],
    [16, 2, 4, 198]
])

y = np.array([
    1, 1, 1, 0, 1,
    0, 1, 1, 0, 0
])

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

np.random.seed(0)

weights = np.random.rand(4)
bias = np.random.rand()

alpha = 0.1
epochs = 1000

for epoch in range(epochs):
    for i in range(len(X)):
        net = np.dot(X[i], weights) + bias
        output = sigmoid(net)
        error = y[i] - output

        weights += alpha * error * X[i]
        bias += alpha * error

print("Final Weights")
print(weights)
print("\nBias")
print(bias)
print("\nPredictions")

for i in range(len(X)):
    net = np.dot(X[i], weights) + bias
    output = sigmoid(net)
    prediction = 1 if output >= 0.5 else 0
    print(
        "Customer", i + 1,
        "Predicted =", prediction,
        "Actual =", y[i]
    )