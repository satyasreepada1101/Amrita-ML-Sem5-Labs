# A2

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def get_data(data):
    X = data[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]]
    y = data["Payment (Rs)"].apply(lambda x: "rich" if x > 200 else "poor")
    return X, y

def train_classifier(X, y):
    model = DecisionTreeClassifier()
    model.fit(X, y)
    return model

data = pd.read_excel("Lab_Session_Data.xlsx", sheet_name="Purchase data")

X, y = get_data(data)

model = train_classifier(X, y)

predictions = model.predict(X)

print("actual class:")
print(y.tolist())

print("\npredicted class:")
print(predictions.tolist())
