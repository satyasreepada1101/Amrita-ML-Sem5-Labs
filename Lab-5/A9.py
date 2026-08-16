import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from A7 import newKNN
from A2 import weighted_knn

df = pd.read_csv("features.csv")
df = df[df["person_id"].isin(["A", "B"])]

X = df.drop( ["person_id", "image_name"], axis=1 )
y = df["person_id"]

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=42 )

k_values = [1, 3, 5]

normal_acc = []
weighted_acc = []

for k in k_values:

    # normal knn
    model = newKNN(k)
    model.fit(X_train, y_train)
    normal_acc.append( model.score(X_test, y_test) )

    # weighted knn
    correct = 0
    for i in range(len(X_test)):
        pred = weighted_knn( X_train, y_train, X_test.iloc[i], k )
        if pred == y_test.iloc[i]:
            correct += 1
    acc = correct / len(y_test)
    weighted_acc.append(acc)


plt.plot( k_values, normal_acc, marker="o" )
plt.plot( k_values, weighted_acc, marker="o" )
plt.xlabel("k")
plt.ylabel("accuracy")
plt.legend( ["normal knn", "weighted knn"] )
plt.show()