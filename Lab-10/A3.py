
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("features.csv")

# Separate features and target
X = data.drop(columns=["person_id", "image_name"])
y = data["person_id"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Classification models
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC()
}

# Evaluate models using original features
original_results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    original_results[name] = accuracy_score(y_test, predictions) * 100

# Apply PCA, retaining 99% explained variance
pca = PCA(n_components=0.99)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("Original number of features:", X_train.shape[1])
print("Number of PCA components:", pca.n_components_)
print("Explained variance retained:",
      round(sum(pca.explained_variance_ratio_) * 100, 2), "%")

# Evaluate models using PCA-transformed features
pca_results = {}

for name, model in models.items():
    model.fit(X_train_pca, y_train)
    predictions = model.predict(X_test_pca)
    pca_results[name] = accuracy_score(y_test, predictions) * 100

# Compare results
results = pd.DataFrame({
    "Model": list(models.keys()),
    "Original Features Accuracy (%)": [
        original_results[name] for name in models
    ],
    "PCA Accuracy (%)": [
        pca_results[name] for name in models
    ]
})

print("\nClassification Performance Comparison:")
print(results.round(2).to_string(index=False))

# Plot cumulative explained variance
plt.figure(figsize=(8, 5))

cumulative_variance = pca.explained_variance_ratio_.cumsum()

plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance * 100,
    marker="o"
)

plt.axhline(y=99, linestyle="--", label="99% variance")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance (%)")
plt.title("PCA - Cumulative Explained Variance")
plt.legend()
plt.tight_layout()
plt.savefig("A3_pca_variance.png", dpi=300)
plt.show()

# Plot accuracy comparison
results_plot = results.set_index("Model")

results_plot.plot(kind="bar", figsize=(9, 5))
plt.ylabel("Accuracy (%)")
plt.title("Classification Accuracy: Original Features vs PCA")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("A3_accuracy_comparison.png", dpi=300)
plt.show()
