import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("features.csv")

# Create feature matrix X
X = data.drop(columns=["person_id", "image_name"])

# Create target y
y = data["person_id"]

# Split into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

# Create kNN classifier with k = 3
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn.fit(X_train, y_train)

# Predict test samples
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Display predictions
print("Predicted values:")
print(y_pred)

# Display actual values
print("\nActual values:")
print(y_test.values)

# Display accuracy
print("\nAccuracy:", accuracy)