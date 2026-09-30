from sklearn.neural_network import MLPClassifier
import numpy as np

X_and = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

y_and = np.array([0,0,0,1])

and_model = MLPClassifier(
    hidden_layer_sizes=(4,),
    activation='logistic',
    solver='lbfgs',
    max_iter=5000,
    random_state=42
)

and_model.fit(X_and, y_and)

print("AND Gate Predictions:")
print(and_model.predict(X_and))

X_xor = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

y_xor = np.array([0,1,1,0])

xor_model = MLPClassifier(
    hidden_layer_sizes=(4,),
    activation='logistic',
    solver='lbfgs',
    max_iter=5000,
    random_state=42
)

xor_model.fit(X_xor, y_xor)

print("\nXOR Gate Predictions:")
print(xor_model.predict(X_xor))