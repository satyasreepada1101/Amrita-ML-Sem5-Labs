import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("features.csv")

# select 2 features
X = df[["top_left_R_mean", "bottom_right_B_var"]]

# target column
y = df["person_id"]

# converting class labels to numbers - label encoding
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# creating decision tree
model = DecisionTreeClassifier(max_depth=3)

# training model
model.fit(X_train, y_train)

# creating mesh grid
x_min = X.iloc[:, 0].min() - 1
x_max = X.iloc[:, 0].max() + 1

y_min = X.iloc[:, 1].min() - 1
y_max = X.iloc[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 1),
    np.arange(y_min, y_max, 1)
)

# creating dataframe for prediction
grid_points = pd.DataFrame({
    "top_left_R_mean": xx.ravel(),
    "bottom_right_B_var": yy.ravel()
})

# predicting
Z = model.predict(grid_points)

# reshaping output
Z = Z.reshape(xx.shape)

# plotting decision boundary
plt.figure(figsize=(10, 6))

plt.contourf(xx, yy, Z, alpha=0.4)

# plotting actual data points
plt.scatter(
    X.iloc[:, 0],
    X.iloc[:, 1],
    c=y,
    edgecolors="black"
)

plt.xlabel("top_left_R_mean")
plt.ylabel("bottom_right_B_var")

plt.title("Decision Boundary using Decision Tree")

plt.show()