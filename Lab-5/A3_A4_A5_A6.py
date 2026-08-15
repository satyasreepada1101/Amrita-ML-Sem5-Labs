import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("features.csv")
df = df[df["person_id"].isin(["A", "B"])]

X = df.drop( ["person_id", "image_name"], axis=1 )
y = df["person_id"]

# a3 - dataset into test and train

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=42 )
print("\ntraining samples:", len(X_train))
print("testing samples:", len(X_test))

# a4 - train the train set with k=3

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
print("\nmodel trained")

# a5 - testing accuracy using knn model

accuracy = knn.score( X_test, y_test )
print("\naccuracy:", accuracy)

# a6 - prediction behavior for test data

predictions = knn.predict(X_test)
print("\npredicted labels:")
print(predictions)
print("\nactual labels:")
print(y_test.values)
