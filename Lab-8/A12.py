import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("features.csv")

encoder = LabelEncoder()
y = encoder.fit_transform(df["person_id"])

X = df.select_dtypes(include=["number"])

if "person_id" in X.columns:
    X = X.drop("person_id", axis=1)

print("Features used:", len(X.columns))

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = MLPClassifier(
    hidden_layer_sizes=(100,),
    activation="relu",
    solver="adam",
    max_iter=2000,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy =", round(accuracy * 100, 2), "%")
print("\nClassification Report")
print(classification_report(y_test, y_pred))