import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([[0], [1], [1], [0]])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

np.random.seed(0)

w_hidden = np.random.uniform(-1, 1, (2, 2))
w_output = np.random.uniform(-1, 1, (2, 1))

b_hidden = np.random.uniform(-1, 1, (1, 2))
b_output = np.random.uniform(-1, 1, (1, 1))

learning_rate = 0.05
errors = []

for epoch in range(1000):
    hidden_input = np.dot(X, w_hidden) + b_hidden
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, w_output) + b_output
    final_output = sigmoid(final_input)

    error = y - final_output
    sse = np.sum(error ** 2)

    errors.append(sse)

    if sse <= 0.002:
        break

    d_output = error * sigmoid_derivative(final_output)
    hidden_error = np.dot(d_output, w_output.T)
    d_hidden = hidden_error * sigmoid_derivative(hidden_output)

    w_output += learning_rate * np.dot(hidden_output.T, d_output)
    b_output += learning_rate * np.sum(d_output, axis=0, keepdims=True)

    w_hidden += learning_rate * np.dot(X.T, d_hidden)
    b_hidden += learning_rate * np.sum(d_hidden, axis=0, keepdims=True)

print("Epochs =", epoch + 1)
print("\nNetwork Output:")
print(np.round(final_output, 4))

plt.plot(range(1, len(errors)+1), errors)
plt.xlabel("Epoch")
plt.ylabel("SSE")
plt.title("Backpropagation XOR Gate")
plt.grid(True)
plt.show()