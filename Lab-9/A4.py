import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# Load dataset
def load_dataset(file_path):

    df = pd.read_csv(file_path)

    X = df.drop(columns=["person_id", "image_name"])
    y = df["person_id"]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    return X, y


# Split dataset
def split_dataset(X, y):

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


# Create pipeline
def create_pipeline():

    pipeline = Pipeline([
        ("scaler", StandardScaler()),

        ("classifier",
         RandomForestClassifier(
             n_estimators=100,
             random_state=42
         ))
    ])

    return pipeline


# Train model
def train_pipeline(model, X_train, y_train):

    model.fit(X_train, y_train)

    return model


# Evaluate model
def evaluate_pipeline(model, X_test, y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        zero_division=0
    )

    return accuracy, report


# Main Program

X, y = load_dataset("features.csv")

X_train, X_test, y_train, y_test = split_dataset(X, y)

pipeline_model = create_pipeline()

pipeline_model = train_pipeline(
    pipeline_model,
    X_train,
    y_train
)

accuracy, report = evaluate_pipeline(
    pipeline_model,
    X_test,
    y_test
)

print("================================")
print("PIPELINE RESULTS")
print("================================")

print("\nAccuracy:")
print(accuracy)

print("\nClassification Report:")
print(report)