import pandas as pd
import joblib
import xgboost as xgb

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report


# Load dataset
df = pd.read_csv("data/processed/features.csv")

# Features used for training
FEATURES = [
    "rpm",
    "speed_kmh",
    "clutch_pct",
    "throttle_pct",
    "brake_flag",
    "incline_deg",
    "rpm_rate",
    "clutch_variance",
    "brake_toggles",
    "rpm_rolling_mean_5",
    "rpm_rolling_std_5",
    "rpm_x_clutch",
    "brake_x_clutch",
    "embedded_risk_score",
]

# -----------------------------
# SESSION-BASED SPLIT (IMPORTANT)
# -----------------------------
import numpy as np
sessions = np.array(df["session_id"].unique())

train_sessions, test_sessions = train_test_split(
    sessions,
    test_size=0.2,
    random_state=42
)

train_df = df[df["session_id"].isin(train_sessions)]
test_df = df[df["session_id"].isin(test_sessions)]

X_train = train_df[FEATURES]
y_train = train_df["stall_risk_label"]

X_test = test_df[FEATURES]
y_test = test_df["stall_risk_label"]

# -----------------------------
# XGBOOST MODEL
# -----------------------------
model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
preds = model.predict(X_test)

print("\nClassification Report:\n")
print(
    classification_report(
        y_test,
        preds,
        target_names=["Safe", "Warning", "High Risk"]
    )
)

# -----------------------------
# CROSS VALIDATION (GENERALIZATION CHECK)
# -----------------------------
sample_df = df.sample(50000, random_state=42)

cv_scores = cross_val_score(
    model,
    sample_df[FEATURES],
    sample_df["stall_risk_label"],
    cv=3
)
print("\nCross Validation Accuracy:")
print(f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# -----------------------------
# SAVE MODEL
# -----------------------------
joblib.dump(model, "models/xgboost.pkl")

# Save test set for SHAP later
X_test.to_csv("data/processed/X_test.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("\nModel saved successfully.")