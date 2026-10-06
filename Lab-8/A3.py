import numpy as np

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 0, 0, 1])

def bipolar_step(x):
    if x > 0:
        return 1
    elif x == 0:
        return 0
    else:
        return -1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return max(0, x)

def train_and_gate(activation_name):
    w0 = 10
    w1 = 0.2
    w2 = -0.75
    alpha = 0.05
    max_epochs = 1000
    convergence_error = 0.002
    for epoch in range(max_epochs):
        sse = 0
        for i in range(len(X)):
            x1 = X[i][0]
            x2 = X[i][1]
            net = w0 + w1*x1 + w2*x2
            if activation_name == "Bipolar Step":
                output = bipolar_step(net)
                if output == -1:
                    output = 0
            elif activation_name == "Sigmoid":
                output = sigmoid(net)
                if output >= 0.5:
                    output = 1
                else:
                    output = 0
            elif activation_name == "ReLU":
                output = relu(net)
                if output > 0:
                    output = 1
                else:
                    output = 0
            error = y[i] - output
            sse += error**2
            w0 += alpha * error
            w1 += alpha * error * x1
            w2 += alpha * error * x2
        if sse <= convergence_error:
            return (activation_name, epoch + 1, w0, w1, w2)
    return (activation_name, max_epochs, w0, w1, w2)

activations = ["Bipolar Step", "Sigmoid", "ReLU"]

for act in activations:
    result = train_and_gate(act)
    print("\nActivation :", result[0])
    print("Epochs:", result[1])
    print("w0:", result[2])
    print("w1:", result[3])
    print("w2:", result[4])