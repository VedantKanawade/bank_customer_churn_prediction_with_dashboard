import joblib
import pandas as pd
import matplotlib.pyplot as plt


MODEL_PATH = "../models/random_forest_model.pkl"
DATA_PATH = "../data/processed/engineered_data.csv"


# Load model
rf_model = joblib.load(MODEL_PATH)

# Load data to get feature names
df = pd.read_csv(DATA_PATH)

# Drop leakage + target
drop_cols = ["Exited", "Complain", "Risk_Score"]
X = df.drop(columns=drop_cols)


# Extract importance
importances = rf_model.feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)


print("\nTop 15 Important Features:")
print(feature_importance_df.head(15))


# Plot
plt.figure(figsize=(10,6))

plt.barh(
    feature_importance_df["Feature"].head(15)[::-1],
    feature_importance_df["Importance"].head(15)[::-1]
)

plt.title("Top 15 Churn Drivers - Random Forest")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()
