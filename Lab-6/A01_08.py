import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from A01_07 import KNNClassifier

data = pd.read_csv("features.csv")

X = data.drop(columns=["person_id", "image_name"])

y = data["person_id"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

k_values = [1, 3, 5, 7, 9]

my_knn_accuracy = []
sklearn_knn_accuracy = []

for k in k_values:
    my_knn = KNNClassifier(k=k)

    my_knn.fit(X_train, y_train)

    my_accuracy = my_knn.score(X_test, y_test)

    my_knn_accuracy.append(my_accuracy)

    sklearn_knn = KNeighborsClassifier(n_neighbors=k)

    sklearn_knn.fit(X_train, y_train)

    predictions = sklearn_knn.predict(X_test)

    sklearn_accuracy = (
        predictions == y_test.values
    ).mean()

    sklearn_knn_accuracy.append(sklearn_accuracy)

print("Accuracy Comparison")
print("-------------------")

for i in range(len(k_values)):

    print(
        "k =", k_values[i],
        "| My KNN =", my_knn_accuracy[i],
        "| Scikit-learn KNN =", sklearn_knn_accuracy[i]
    )

plt.plot(
    k_values,
    my_knn_accuracy,
    marker="o",
    label="My Custom KNN"
)

plt.plot(
    k_values,
    sklearn_knn_accuracy,
    marker="o",
    label="Scikit-learn KNN"
)

plt.xlabel("Value of k")
plt.ylabel("Accuracy")
plt.title("Custom KNN vs Scikit-learn KNN")
plt.xticks(k_values)
plt.legend()
plt.grid()
plt.show()