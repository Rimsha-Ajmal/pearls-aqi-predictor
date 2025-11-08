# compute_features_historical.py

import pandas as pd
import numpy as np

# ---------------- CONFIG ----------------
# Focused high-impact features for 3-day AQI prediction
TOP_FEATURES = [
    'aqi_t_plus_12', 'aqi_t_plus_24', 'aqi_t_plus_48', 'aqi_t_plus_72',  # 3-day ahead target
    'temp_roll_mean_24', 'month', 'pm2_5_roll_mean_24', 'wind_speed_roll_mean_24',
    'pm2_5_roll_std_24', 'humidity_roll_mean_24', 'pm10_lag_24', 'wind_speed_roll_std_24',
    'pm2_5_roll_mean_12', 'o3_roll_std_24', 'no2_roll_std_24', 'co_lag_24', 'humidity_roll_std_24',
    'no2_lag_24', 'so2_roll_mean_24', 'no2_roll_mean_24', 'dayofweek', 'o3_lag_24', 'so2_roll_std_24',
    'no2_roll_mean_12', 'o3_roll_mean_24', 'co_roll_mean_24', 'pm10_roll_mean_24', 'so2_roll_mean_12',
    'pm10_roll_std_12', 'temp_roll_mean_12', 'pm2_5_lag_24', 'co_roll_std_24', 'temp_lag_6',
    'temp_roll_std_24', 'co_lag_1', 'so2_lag_24', 'pm10_roll_std_24', 'co_lag_12', 'co',
    'aqi_roll_std_24', 'pm2_5_lag_12', 'pm10_lag_12', 'co_roll_mean_12', 'co_roll_std_3', 'humidity'
]


def compute_features_and_targets(
    df,
    datetime_col="datetime",
    value_cols=None,
    lags=[1, 3, 6, 12, 24],
    rolling_windows=[3, 6, 12, 24],
    horizon_hours=[1, 6, 12, 24, 48, 72]  # include 3-day prediction
):
    if value_cols is None:
        value_cols = ["aqi", "pm2_5", "pm10", "co", "no2", "so2", "o3",
                      "temp", "humidity", "wind_speed"]

    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce", utc=True)
    df = df.sort_values(datetime_col).reset_index(drop=True)
    df.set_index(datetime_col, inplace=True)

    for col in value_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Time & cyclic features
    df["hour"], df["dayofweek"], df["month"] = df.index.hour, df.index.dayofweek, df.index.month
    df["hour_sin"], df["hour_cos"] = np.sin(2*np.pi*df["hour"]/24), np.cos(2*np.pi*df["hour"]/24)
    df["dow_sin"], df["dow_cos"] = np.sin(2*np.pi*df["dayofweek"]/7), np.cos(2*np.pi*df["dayofweek"]/7)
    df["month_sin"], df["month_cos"] = np.sin(2*np.pi*(df["month"]-1)/12), np.cos(2*np.pi*(df["month"]-1)/12)

    # Lag features
    lag_features = {f"{col}_lag_{l}": df[col].shift(l) for col in value_cols for l in lags}
    lag_df = pd.DataFrame(lag_features, index=df.index)

    # Rolling features
    roll_features = {}
    for w in rolling_windows:
        roll_mean = df[value_cols].rolling(window=w, min_periods=1).mean().shift(1)
        roll_std = df[value_cols].rolling(window=w, min_periods=1).std().shift(1).fillna(0)
        for col in value_cols:
            roll_features[f"{col}_roll_mean_{w}"] = roll_mean[col]
            roll_features[f"{col}_roll_std_{w}"] = roll_std[col]
    roll_df = pd.DataFrame(roll_features, index=df.index)

    # Future targets
    target_df = pd.DataFrame({f"aqi_t_plus_{h}": df["aqi"].shift(-h) for h in horizon_hours}, index=df.index)

    # Merge all features
    df = pd.concat([df, lag_df, roll_df, target_df], axis=1).reset_index()
    df = df.dropna(subset=["aqi"]).reset_index(drop=True)

    # Keep only top high-impact features
    keep_cols = ["datetime"] + [c for c in TOP_FEATURES if c in df.columns]
    # Include targets if missing
    for h in horizon_hours:
        target_col = f"aqi_t_plus_{h}"
        if target_col not in keep_cols and target_col in df.columns:
            keep_cols.append(target_col)
    df = df[keep_cols]

    return df


# ---------------- MAIN -----------------
if __name__ == "__main__":
    input_path = "historical_data.csv"
    df_raw = pd.read_csv(input_path)
    print(f"✅ Raw data loaded: {input_path} -> {len(df_raw)} rows")

    df_features = compute_features_and_targets(df_raw)
    print(f"✅ Features computed: {df_features.shape}")
    print(df_features.head())

    df_features.to_csv("computed_features_historical_selected.csv", index=False)
    print(f"✅ Feature dataset saved as computed_features_historical_selected.csv")
