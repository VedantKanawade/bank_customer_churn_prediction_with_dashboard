import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Paths
MODEL_PATH = "../models/xgboost_model.pkl"
DATA_PATH = "../data/processed/engineered_data.csv"

# Load model and data
xgb_model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

# Drop leakage / target columns
drop_cols = ["Exited", "Complain", "Risk_Score"]
X = df.drop(columns=drop_cols)
y = df["Exited"]

# SHAP explainer
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X)

# Summary plot - global feature importance
plt.figure()
shap.summary_plot(shap_values, X, plot_type="bar", show=True)

# Detailed summary plot - feature effects
plt.figure()
shap.summary_plot(shap_values, X, show=True)
