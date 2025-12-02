import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------
# 1️⃣ Create metrics DataFrame
# -----------------------------
data = {
    "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
    "ROC-AUC": [0.778, 0.870, 0.860],
    "Churn Recall": [0.64, 0.62, 0.66],
    "Accuracy": [0.78, 0.85, 0.84]
}

df = pd.DataFrame(data)
df.set_index("Model", inplace=True)

# -----------------------------
# 2️⃣ Set Seaborn style
# -----------------------------
sns.set(style="whitegrid")

# -----------------------------
# 3️⃣ Plot heatmap
# -----------------------------
plt.figure(figsize=(8,3))
sns.heatmap(
    df,
    annot=True,          # Show numbers in cells
    cmap="YlGnBu",       # Color palette
    fmt=".2f",           # Format numbers
    cbar=True,           # Show color bar
    linewidths=0.5,      # Cell borders
    linecolor='white'
)

plt.title("ML Model Performance Heatmap", fontsize=14, fontweight='bold')
plt.yticks(rotation=0)    # Keep model names horizontal
plt.xticks(rotation=45)   # Rotate metric names

plt.tight_layout()
plt.show()