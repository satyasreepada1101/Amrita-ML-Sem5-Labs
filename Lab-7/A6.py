import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split

df = pd.read_csv("features.csv")

X = df.drop(["person_id", "image_name"], axis=1)

# target column - y
y = df["person_id"]

# splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

plt.figure(figsize=(15, 10))

plot_tree(model, feature_names=X.columns, filled=True)

plt.title("Decision Tree")
plt.show()