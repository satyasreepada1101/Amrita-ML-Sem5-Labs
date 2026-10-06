import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from lime.lime_tabular import LimeTabularExplainer


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


# Train Random Forest model
def train_model(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


# Evaluate model
def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    return accuracy


# Generate LIME explanation
def explain_prediction(model, X_train, X_test):

    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=[str(i) for i in model.classes_],
        mode="classification"
    )

    sample_index = 0

    explanation = explainer.explain_instance(
        X_test.iloc[sample_index].values,
        model.predict_proba,
        num_features=10
    )

    print("\nLIME Explanation:")
    print(explanation.as_list())

    explanation.save_to_file(
        "lime_explanation.html"
    )

    print("\nExplanation saved as lime_explanation.html")


# Main Program

X, y = load_dataset("features.csv")

X_train, X_test, y_train, y_test = split_dataset(
    X,
    y
)

model = train_model(
    X_train,
    y_train
)

accuracy = evaluate_model(
    model,
    X_test,
    y_test
)

print("Accuracy:", accuracy)

explain_prediction(
    model,
    X_train,
    X_test
)