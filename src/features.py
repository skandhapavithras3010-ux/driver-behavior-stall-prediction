import pandas as pd


def add_features(df):

    # Rolling RPM behaviour
    df["rpm_rolling_mean_5"] = (
        df.groupby("session_id")["rpm"]
        .transform(
            lambda x: x.rolling(
                5,
                min_periods=1
            ).mean()
        )
    )

    df["rpm_rolling_std_5"] = (
        df.groupby("session_id")["rpm"]
        .transform(
            lambda x: x.rolling(
                5,
                min_periods=1
            ).std()
        )
        .fillna(0)
    )

    # Interaction features

    df["rpm_x_clutch"] = (
        df["rpm"]
        * df["clutch_pct"]
        / 100
    )

    df["brake_x_clutch"] = (
        df["brake_flag"]
        * df["clutch_pct"]
    )

    df["speed_deficit"] = (
        15 - df["speed_kmh"]
    )

    # Embedded-system-inspired risk proxy

    df["embedded_risk_score"] = (
        0.40 * (1 - df["rpm"] / 5000)
        + 0.30 * (df["clutch_variance"] / 100)
        + 0.20 * df["brake_flag"]
        + 0.10 * (df["incline_deg"] / 15)
    )

    return df


df = pd.read_csv(
    "data/raw/telemetry.csv"
)

df = add_features(df)

df.to_csv(
    "data/processed/features.csv",
    index=False
)

print("\nFeature engineering complete")
print("Shape:", df.shape)

print("\nNew Features Added:")

new_features = [
    "rpm_rolling_mean_5",
    "rpm_rolling_std_5",
    "rpm_x_clutch",
    "brake_x_clutch",
    "speed_deficit",
    "embedded_risk_score"
]

for feature in new_features:
    print("-", feature)