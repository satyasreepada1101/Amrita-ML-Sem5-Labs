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

X_bias = np.c_[np.ones(len(X)), X]
weights = np.linalg.pinv(X_bias) @ y

print("Weights:")
print(weights)
print("\nPredictions:")

correct = 0

for i in range(len(X_bias)):
    output = np.dot(X_bias[i], weights)
    prediction = 1 if output >= 0.5 else 0

    if prediction == y[i]:
        correct += 1

    print(
        "Customer", i + 1,
        "Predicted =", prediction,
        "Actual =", y[i]
    )

accuracy = (correct / len(y)) * 100

print("\nAccuracy =", accuracy, "%")