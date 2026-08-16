import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from A7 import newKNN

df = pd.read_csv("features.csv")
df = df[df["person_id"].isin(["A", "B"])]

X = df.drop( ["person_id", "image_name"], axis=1 )
y = df["person_id"]

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=42 )

k_values = [1, 3, 5]

my_acc = []
sklearn_acc = []

for k in k_values:

    # my knn
    my_model = newKNN(k)
    my_model.fit(X_train, y_train)
    my_acc.append( my_model.score(X_test, y_test) )

    # sklearn knn
    sk_model = KNeighborsClassifier( n_neighbors=k )
    sk_model.fit(X_train, y_train)
    sklearn_acc.append( sk_model.score(X_test, y_test) )

plt.plot(k_values, my_acc, marker="o")
plt.plot(k_values, sklearn_acc, marker="o")
plt.xlabel("k")
plt.ylabel("accuracy")
plt.legend( ["my knn", "sklearn knn"] )
plt.show()