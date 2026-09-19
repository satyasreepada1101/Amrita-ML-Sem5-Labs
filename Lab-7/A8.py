import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("features.csv")

X = df.drop(["person_id", "image_name"], axis=1)

y = df["person_id"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = DecisionTreeClassifier()

# hyperparameters to test
parameters = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3, 5, 7, 10],
    "min_samples_split": [2, 5, 10]
}

# grid search
grid_search = GridSearchCV(
    estimator=model,
    param_grid=parameters,
    cv=5,
    scoring="accuracy"
)

# training
grid_search.fit(X_train, y_train)

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Accuracy:")
print(grid_search.best_score_)