# -*- coding: utf-8 -*-
"""
compute_features_realtime.py

Compute derived features from real-time data and insert into model_features Feature Group in Hopsworks.
"""

import hopsworks
import pandas as pd
import numpy as np
import os

def compute_features_and_targets(
    df,
    datetime_col="datetime",
    value_cols=None,
    lags=[1, 3, 6, 12, 24],
    rolling_windows=[3, 6, 12, 24],
    horizon_hours=[1, 6, 12, 24, 48, 72]
):
    """
    Compute lag, rolling, cyclic time, and target features for AQI + weather data.
    Returns a feature-rich DataFrame ready for ML or feature store ingestion.
    """
    if value_cols is None:
        value_cols = ["aqi", "pm2_5", "pm10", "co", "no2", "so2", "o3",
                      "temp", "humidity", "wind_speed"]

    df = df.copy()

    # Parse datetime and sort
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce", utc=True)
    df = df.sort_values(datetime_col).reset_index(drop=True)
    df.set_index(datetime_col, inplace=True)

    # Ensure numeric
    for col in value_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Time features ---
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month

    # Cyclic encodings
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)

    # --- AQI derived simple features ---
    if "aqi" in df.columns:
        df["aqi_diff_1"] = df["aqi"].diff(1)
        df["aqi_pct_change"] = (
            df["aqi"].pct_change()
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )

    # --- Lag features ---
    lag_features = []
    for col in value_cols:
        if col not in df.columns:
            continue
        lag_data = {f"{col}_lag_{l}": df[col].shift(l) for l in lags}
        lag_features.append(pd.DataFrame(lag_data, index=df.index))
    if lag_features:
        df = pd.concat([df] + lag_features, axis=1)

    # --- Rolling mean & std ---
    roll_feature_frames = []
    for w in rolling_windows:
        roll_mean = df[value_cols].rolling(window=w, min_periods=1).mean().shift(1)
        roll_std = df[value_cols].rolling(window=w, min_periods=1).std().shift(1).fillna(0)
        roll_mean.columns = [f"{col}_roll_mean_{w}" for col in value_cols]
        roll_std.columns = [f"{col}_roll_std_{w}" for col in value_cols]
        roll_feature_frames.append(pd.concat([roll_mean, roll_std], axis=1))
    if roll_feature_frames:
        df = pd.concat([df] + roll_feature_frames, axis=1).copy()

    # --- Future targets ---
    if "aqi" in df.columns:
        for h in horizon_hours:
            df[f"aqi_t_plus_{h}"] = df["aqi"].shift(-h)

    # --- Cleanup ---
    df = df.reset_index()
    df = df.dropna(axis=1, how="all")

    # Drop constant columns except AQI and targets
    nunique = df.nunique(dropna=True)
    const_cols = nunique[nunique <= 1].index.tolist()
    targets = [c for c in df.columns if c.startswith("aqi_t_plus_")]
    keep_cols = set(["aqi"] + targets)
    cols_to_drop = [c for c in const_cols if c not in keep_cols]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    # Drop zero-only columns
    zero_cols = [c for c in df.columns if (pd.to_numeric(df[c], errors="coerce").fillna(0) == 0).all()]
    zero_cols = [c for c in zero_cols if c not in keep_cols]
    df = df.drop(columns=zero_cols, errors="ignore")

    # Drop rows with missing AQI
    if "aqi" in df.columns:
        df = df.dropna(subset=["aqi"]).reset_index(drop=True)

    return df


def main():
    # --- 1️⃣ Login using API key from environment variable ---
    api_key = os.environ.get("HOPSWORKS_API_KEY")
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()

    raw_fg = fs.get_feature_group("raw_observations", version=2)
    feature_fg = fs.get_feature_group("model_features", version=1)

    # --- 2️⃣ Fetch recent raw observations ---
    df_recent = raw_fg.read().sort_values("datetime")

    # --- 3️⃣ Compute features ---
    df_recent_features = compute_features_and_targets(df_recent)

    # --- 4️⃣ Align with historical schema ---
    feature_schema = {f.name: f.type for f in feature_fg.features}
    for col in feature_schema:
        if col not in df_recent_features.columns:
            df_recent_features[col] = np.nan
    df_recent_features = df_recent_features[list(feature_schema.keys())]

    # --- 5️⃣ Fill NaNs for integer columns ---
    int_columns_with_nan = ["humidity", "month"]
    for col in int_columns_with_nan:
        if col in df_recent_features.columns:
            df_recent_features[col] = df_recent_features[col].fillna(-1).astype(int)

    # --- 6️⃣ Cast remaining columns according to schema ---
    for col, dtype in feature_schema.items():
        if col in int_columns_with_nan:
            continue
        if dtype == "double":
            df_recent_features[col] = df_recent_features[col].astype(float)
        elif dtype in ["int", "bigint"]:
            df_recent_features[col] = df_recent_features[col].astype(float)

    # --- 7️⃣ Insert into model_features ---
    feature_fg.insert(df_recent_features)
    print("✅ Computed features from raw_observations inserted into model_features successfully!")

    # --- 8️⃣ Optional: verify last 10 rows ---
    df_check = feature_fg.read().sort_values("datetime").tail(10)
    print(df_check)


if __name__ == "__main__":
    main()
