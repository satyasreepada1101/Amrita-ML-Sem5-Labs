import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("features.csv")

# Select numerical feature columns
X = data.drop(columns=["person_id", "image_name"])

# Calculate feature correlations
correlation_matrix = X.corr()

# Display correlation matrix
print("Feature Correlation Matrix:")
print(correlation_matrix.round(2))

# Plot heatmap
plt.figure(figsize=(14, 10))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("A2_correlation_heatmap.png", dpi=300)
plt.show()
