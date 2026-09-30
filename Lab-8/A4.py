import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 0, 0, 1])

def step_activation(x):
    return 1 if x >= 0 else 0

def train_perceptron(alpha):
    w0 = 10
    w1 = 0.2
    w2 = -0.75
    max_epochs = 1000
    for epoch in range(max_epochs):
        sse = 0
        for i in range(len(X)):
            x1 = X[i][0]
            x2 = X[i][1]
            net = w0 + w1 * x1 + w2 * x2
            output = step_activation(net)
            error = y[i] - output
            sse += error ** 2
            w0 += alpha * error
            w1 += alpha * error * x1
            w2 += alpha * error * x2
        if sse <= 0.002:
            return epoch + 1
    return max_epochs

learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5,
                  0.6, 0.7, 0.8, 0.9, 1.0]

epochs_list = []

for lr in learning_rates:
    epochs = train_perceptron(lr)
    epochs_list.append(epochs)
    print("Learning Rate =", lr,
          " Epochs =", epochs)

plt.plot(learning_rates, epochs_list, marker='o')
plt.xlabel("Learning Rate")
plt.ylabel("Iterations to Converge")
plt.title("Learning Rate vs Iterations")
plt.grid(True)
plt.show()