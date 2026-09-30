import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 1, 1, 0])

def step_activation(x):
    return 1 if x >= 0 else 0

w0 = 10
w1 = 0.2
w2 = -0.75

alpha = 0.05
max_epochs = 1000
errors_per_epoch = []

for epoch in range(max_epochs):
    sse = 0

    for i in range(len(X)):
        x1 = X[i][0]
        x2 = X[i][1]

        net = w0 + w1*x1 + w2*x2
        output = step_activation(net)
        error = y[i] - output

        sse += error**2

        w0 += alpha * error
        w1 += alpha * error * x1
        w2 += alpha * error * x2

    errors_per_epoch.append(sse)

    if sse <= 0.002:
        break

print("Epochs =", epoch + 1)
print("\nFinal Weights")
print("w0 =", w0)
print("w1 =", w1)
print("w2 =", w2)
print("\nFinal Predictions")

for i in range(len(X)):
    net = w0 + w1*X[i][0] + w2*X[i][1]
    pred = step_activation(net)
    print(X[i], "->", pred)

plt.plot(range(1, len(errors_per_epoch)+1), errors_per_epoch)
plt.xlabel("Epoch")
plt.ylabel("SSE")
plt.title("XOR Gate Learning")
plt.grid(True)
plt.show()