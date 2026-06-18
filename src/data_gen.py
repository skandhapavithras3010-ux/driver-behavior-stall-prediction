import numpy as np
import pandas as pd


def get_driver_type(driver_num):
    if driver_num <= 12:
        return "smooth"
    elif driver_num <= 25:
        return "cautious"
    elif driver_num <= 37:
        return "aggressive"
    else:
        return "indecisive"


def generate_session(driver_id, session_id, driver_type, n=300, seed=None):

    rng = np.random.default_rng(seed)
    t = np.arange(n)

    # --------------------------------------------------
    # Driver Behaviour Profiles
    # --------------------------------------------------

    if driver_type == "smooth":

        rpm = (
            1700
            + 250 * np.sin(t * 0.04)
            + rng.normal(0, 40, n)
        )

        clutch_pct = (
            45
            + 15 * np.sin(t * 0.06)
            + rng.normal(0, 3, n)
        )

        brake_prob = 0.06

    elif driver_type == "cautious":

        rpm = (
            1400
            + 300 * np.sin(t * 0.05)
            + rng.normal(0, 60, n)
        )

        clutch_pct = (
            60
            + 20 * np.sin(t * 0.07)
            + rng.normal(0, 5, n)
        )

        brake_prob = 0.18

    elif driver_type == "aggressive":

        rpm = (
            2000
            + 700 * np.sin(t * 0.08)
            + rng.normal(0, 120, n)
        )

        for event in rng.choice(n, size=8, replace=False):
            rpm[
                max(0, event - 8):
                min(n, event + 8)
            ] -= rng.uniform(800, 1500)

        clutch_pct = (
            40
            + 35 * np.sin(t * 0.10)
            + rng.normal(0, 10, n)
        )

        brake_prob = 0.10

    else:  # indecisive

        rpm = (
            1500
            + 400 * np.sin(t * 0.05)
            + rng.normal(0, 90, n)
        )

        clutch_pct = (
            50
            + 35 * np.sin(t * 0.12)
            + rng.normal(0, 14, n)
        )

        brake_prob = 0.25

    rpm = np.clip(rpm, 500, 5000)

    throttle_pct = np.clip(
        35
        + 20 * np.cos(t * 0.06)
        + rng.normal(0, 5, n),
        0,
        100,
    )

    speed_kmh = np.clip(
        5
        + 7 * np.sin(t * 0.04)
        + rng.normal(0, 1, n),
        0,
        15,
    )

    brake_flag = (
        rng.random(n) < brake_prob
    ).astype(int)

    incline_deg = rng.uniform(0, 12)

    rpm = rpm.astype(int)

    rpm_rate = np.diff(
        rpm,
        prepend=rpm[0]
    )

    clutch_variance = (
        pd.Series(clutch_pct)
        .rolling(10, min_periods=1)
        .var()
        .fillna(0)
        .values
    )

    brake_toggles = (
        pd.Series(brake_flag)
        .rolling(10, min_periods=1)
        .sum()
        .values
    )

    # --------------------------------------------------
    # Composite Risk Score
    # --------------------------------------------------

    rpm_risk = np.clip(
        (1200 - rpm) / 1200,
        0,
        1
    )

    rpm_drop_risk = np.clip(
        (-rpm_rate) / 300,
        0,
        1
    )

    clutch_risk = (
        (
            (clutch_pct > 30)
            & (clutch_pct < 70)
        ).astype(int)
        * (rpm < 1200).astype(int)
    )

    brake_risk = brake_flag

    incline_risk = incline_deg / 12

    variance_risk = np.clip(
        clutch_variance / 250,
        0,
        1
    )

    risk_score = (
        0.30 * rpm_risk
        + 0.25 * rpm_drop_risk
        + 0.20 * clutch_risk
        + 0.10 * brake_risk
        + 0.05 * incline_risk
        + 0.10 * variance_risk
    )

    return pd.DataFrame({
        "session_id": session_id,
        "driver_id": driver_id,
        "driver_type": driver_type,
        "rpm": rpm,
        "speed_kmh": speed_kmh,
        "clutch_pct": clutch_pct,
        "throttle_pct": throttle_pct,
        "brake_flag": brake_flag,
        "incline_deg": incline_deg,
        "rpm_rate": rpm_rate,
        "clutch_variance": clutch_variance,
        "brake_toggles": brake_toggles,
        "risk_score": risk_score
    })


frames = []

for d in range(1, 51):

    driver_type = get_driver_type(d)

    for s in range(1, 11):

        frames.append(
            generate_session(
                driver_id=f"D{d:03}",
                session_id=f"S{d:03}_{s:02}",
                driver_type=driver_type,
                seed=d * 100 + s
            )
        )

df = pd.concat(
    frames,
    ignore_index=True
)

# --------------------------------------------------
# Final Labels
# --------------------------------------------------

df["stall_risk_label"] = pd.qcut(
    df["risk_score"],
    q=[0, 0.60, 0.85, 1.00],
    labels=[0, 1, 2]
).astype(int)
df.drop(columns=["risk_score"], inplace=True)
df.to_csv(
    "data/raw/telemetry.csv",
    index=False
)

print(f"\nGenerated {len(df):,} rows")

print("\nClass Distribution:")
print(
    df["stall_risk_label"]
    .value_counts(normalize=True)
    .sort_index()
)

print("\nDriver Types:")
print(
    df["driver_type"]
    .value_counts()
)