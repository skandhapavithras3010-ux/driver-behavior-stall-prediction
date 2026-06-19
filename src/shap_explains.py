import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# LOAD MODEL + DATA
# -----------------------------
model = joblib.load("models/xgboost.pkl")
X_test = pd.read_csv("data/processed/X_test.csv")

# Use a sample to speed things up
X_sample = X_test.sample(
    min(5000, len(X_test)),
    random_state=42
)

# -----------------------------
# SHAP EXPLANATIONS
# -----------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# -----------------------------
# HANDLE MULTICLASS SHAP OUTPUT
# -----------------------------
if isinstance(shap_values, list):
    high_risk_shap = shap_values[2]
else:
    # Newer SHAP versions
    high_risk_shap = shap_values[:, :, 2]

# -----------------------------
# 1. GLOBAL FEATURE IMPORTANCE
# -----------------------------
shap.summary_plot(
    high_risk_shap,
    X_sample,
    plot_type="bar",
    show=False
)

plt.tight_layout()
plt.savefig(
    "reports/shap_summary.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

# -----------------------------
# 2. TOP 3 FEATURES
# -----------------------------
mean_shap = np.abs(high_risk_shap).mean(axis=0)

top3 = (
    pd.Series(
        mean_shap,
        index=X_sample.columns
    )
    .sort_values(ascending=False)
    .head(3)
)

print("\nTop 3 High Risk Drivers:\n")
print(top3)

top3.to_csv(
    "reports/top3_features.csv"
)

print("\nSHAP analysis completed successfully.")
print("Saved:")
print("- reports/shap_summary.png")
print("- reports/top3_features.csv")