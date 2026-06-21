from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/features.csv")

# Aggregate driver behaviour

driver_stats = (
    df.groupby("driver_id")
    .agg(
        avg_rpm=("rpm", "mean"),
        stall_rate=("stall_risk_label",
                    lambda x: (x == 2).mean()),
        avg_clutch_var=("clutch_variance", "mean"),
        avg_brake_tog=("brake_toggles", "mean"),
        warning_rate=("stall_risk_label",
                      lambda x: (x >= 1).mean())
    )
    .reset_index()
)

# Scale

scaler = StandardScaler()

X_driver = scaler.fit_transform(
    driver_stats.drop("driver_id", axis=1)
)

# Elbow curve

inertias = []

for k in range(2, 8):

    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    km.fit(X_driver)

    inertias.append(km.inertia_)

plt.figure(figsize=(6, 4))

plt.plot(
    range(2, 8),
    inertias,
    marker="o"
)

plt.title("Elbow Curve")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")

plt.tight_layout()

plt.savefig(
    "reports/elbow.png",
    dpi=150
)

plt.close()

# Final clustering

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

driver_stats["profile"] = kmeans.fit_predict(
    X_driver
)
profile_names = {
    0: "Cautious",
    1: "Indecisive",
    2: "Smooth",
    3: "Aggressive"
}

driver_stats["profile_name"] = (
    driver_stats["profile"]
    .map(profile_names)
)

print("\nCluster Summary:\n")

print(
    driver_stats
    .groupby("profile")
    .agg(
        stall_rate=("stall_rate", "mean"),
        clutch_var=("avg_clutch_var", "mean"),
        brake_tog=("avg_brake_tog", "mean"),
        rpm=("avg_rpm", "mean")
    )
)

driver_stats.to_csv(
    "data/processed/driver_profiles.csv",
    index=False
)

print("\nSaved:")
print("- reports/elbow.png")
print("- data/processed/driver_profiles.csv")