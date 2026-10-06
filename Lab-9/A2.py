import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# Load the dataset
def load_dataset(file_path):

    df = pd.read_csv(file_path)

    X = df.drop(columns=["person_id", "image_name"])
    y = df["person_id"]

    # Convert writer labels into numbers
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    return X, y


# Split data into train and test sets
def split_dataset(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


# Tune Perceptron using RandomizedSearchCV
def tune_perceptron(X_train, y_train):

    model = Perceptron(random_state=42)

    parameters = {
        "penalty": [None, "l1", "l2", "elasticnet"],
        "alpha": [0.0001, 0.001, 0.01, 0.1],
        "max_iter": [500, 1000, 1500, 2000],
        "eta0": [0.001, 0.01, 0.1, 1.0]
    }

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=parameters,
        n_iter=10,
        cv=4,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    return search


# Tune MLP using RandomizedSearchCV
def tune_mlp(X_train, y_train):

    model = MLPClassifier(
        random_state=42
    )

    parameters = {
        "hidden_layer_sizes": [
            (50,),
            (100,),
            (50, 50),
            (100, 50)
        ],
        "activation": [
            "relu",
            "tanh"
        ],
        "alpha": [
            0.0001,
            0.001,
            0.01
        ],
        "solver": [
            "adam",
            "sgd"
        ]
    }

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=parameters,
        n_iter=10,
        cv=4,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    return search


# Calculate test accuracy
def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    return accuracy


# Main Program

X, y = load_dataset("features.csv")

X_train, X_test, y_train, y_test = split_dataset(X, y)

# Perceptron
best_perceptron = tune_perceptron(X_train, y_train)

perceptron_accuracy = evaluate_model(
    best_perceptron.best_estimator_,
    X_test,
    y_test
)

print("====================================")
print("PERCEPTRON RESULTS")
print("====================================")

print("Best Parameters:")
print(best_perceptron.best_params_)

print("\nBest Cross Validation Accuracy:")
print(best_perceptron.best_score_)

print("\nTest Accuracy:")
print(perceptron_accuracy)


# MLP
best_mlp = tune_mlp(X_train, y_train)

mlp_accuracy = evaluate_model(
    best_mlp.best_estimator_,
    X_test,
    y_test
)

print("\n====================================")
print("MLP RESULTS")
print("====================================")

print("Best Parameters:")
print(best_mlp.best_params_)

print("\nBest Cross Validation Accuracy:")
print(best_mlp.best_score_)

print("\nTest Accuracy:")
print(mlp_accuracy)