import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from xgboost import XGBClassifier
from catboost import CatBoostClassifier


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


# Train and evaluate a classifier
def evaluate_classifier(model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    return [accuracy, precision, recall, f1]


# Main Program

X, y = load_dataset("features.csv")

X_train, X_test, y_train, y_test = split_dataset(X, y)

models = {

    "SVM":
        SVC(),

    "Decision Tree":
        DecisionTreeClassifier(random_state=42),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

    "AdaBoost":
        AdaBoostClassifier(
            n_estimators=50,
            random_state=42
        ),

    "Naive Bayes":
        GaussianNB(),

    "XGBoost":
        XGBClassifier(
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42
        ),

    "CatBoost":
        CatBoostClassifier(
            verbose=0,
            random_state=42
        )
}

results = []

for model_name, model in models.items():

    metrics = evaluate_classifier(
        model,
        X_train,
        X_test,
        y_train,
        y_test
    )

    results.append(
        [model_name] + metrics
    )

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)

print("\nCLASSIFIER COMPARISON\n")
print(results_df)

results_df.to_csv(
    "A3_Classifier_Results.csv",
    index=False
)

print("\nResults saved to A3_Classifier_Results.csv")