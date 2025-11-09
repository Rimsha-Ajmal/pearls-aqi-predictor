# ====================== aqi_realtime_features.py ======================

# Install required packages (uncomment if running in Colab or Jupyter)
# !pip install hopsworks==4.2.*
# !pip install confluent-kafka
# !pip install pandas numpy

import hopsworks
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ------------------------- Hopsworks Connection ------------------------- #
def connect_hopsworks():
    """
    Connect to Hopsworks project and return project and feature store objects.
    """
    project = hopsworks.login()  # Ensure your API key / login works
    fs = project.get_feature_store()
    print("✅ Connected to Hopsworks project")
    return project, fs


# ------------------------- Compute Realtime Features ------------------------- #
def compute_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 35 real-time AQI features from raw observations.
    """
    df = df_raw.copy()
    
    # Rolling features
    df['temp_roll_mean_24'] = df['temp'].rolling(24, min_periods=1).mean()
    df['temp_roll_std_24'] = df['temp'].rolling(24, min_periods=1).std().fillna(0)
    df['humidity_roll_mean_24'] = df['humidity'].rolling(24, min_periods=1).mean()
    df['humidity_roll_std_24'] = df['humidity'].rolling(24, min_periods=1).std().fillna(0)
    df['wind_speed_roll_mean_24'] = df['wind_speed'].rolling(24, min_periods=1).mean()
    
    df['pm2_5_roll_mean_12'] = df['pm2_5'].rolling(12, min_periods=1).mean()
    df['pm2_5_roll_std_24'] = df['pm2_5'].rolling(24, min_periods=1).std().fillna(0)
    df['pm10_roll_mean_24'] = df['pm10'].rolling(24, min_periods=1).mean()
    df['pm10_roll_std_12'] = df['pm10'].rolling(12, min_periods=1).std().fillna(0)
    df['co_roll_mean_12'] = df['co'].rolling(12, min_periods=1).mean()
    df['no2_roll_mean_12'] = df['no2'].rolling(12, min_periods=1).mean()
    df['so2_roll_mean_24'] = df['so2'].rolling(24, min_periods=1).mean()
    df['o3_roll_mean_24'] = df['o3'].rolling(24, min_periods=1).mean()
    
    # Lag features
    df['temp_lag_6'] = df['temp'].shift(6).fillna(0)
    df['co_lag_12'] = df['co'].shift(12).fillna(0)
    df['pm10_lag_12'] = df['pm10'].shift(12).fillna(0)
    df['pm10_lag_24'] = df['pm10'].shift(24).fillna(0)
    df['pm2_5_lag_24'] = df['pm2_5'].shift(24).fillna(0)
    df['no2_lag_24'] = df['no2'].shift(24).fillna(0)
    df['co_lag_24'] = df['co'].shift(24).fillna(0)
    df['so2_lag_24'] = df['so2'].shift(24).fillna(0)
    
    # AQI future targets
    df['aqi_t_plus_1'] = df['aqi'].shift(-1).fillna(0)
    df['aqi_t_plus_6'] = df['aqi'].shift(-6).fillna(0)
    df['aqi_t_plus_12'] = df['aqi'].shift(-12).fillna(0)
    df['aqi_t_plus_24'] = df['aqi'].shift(-24).fillna(0)
    df['aqi_t_plus_48'] = df['aqi'].shift(-48).fillna(0)
    df['aqi_t_plus_72'] = df['aqi'].shift(-72).fillna(0)
    
    # AQI rolling std
    df['aqi_roll_std_24'] = df['aqi'].rolling(24, min_periods=1).std().fillna(0)
    
    # Day of week
    df['dayofweek'] = df['datetime'].dt.dayofweek
    
    # Additional pollutant std/mean
    df['no2_roll_std_24'] = df['no2'].rolling(24, min_periods=1).std().fillna(0)
    df['co_roll_mean_24'] = df['co'].rolling(24, min_periods=1).mean()
    df['so2_roll_std_24'] = df['so2'].rolling(24, min_periods=1).std().fillna(0)
    df['o3_roll_std_24'] = df['o3'].rolling(24, min_periods=1).std().fillna(0)
    
    # Keep only 35 features in the same order
    feature_cols = [
        'datetime', 'aqi_t_plus_48', 'aqi_t_plus_72', 'temp_roll_mean_24',
        'humidity_roll_std_24', 'pm10_roll_std_12', 'co_roll_mean_12',
        'wind_speed_roll_mean_24', 'co_lag_12', 'temp_lag_6', 'temp_roll_std_24',
        'pm2_5_roll_mean_12', 'no2_roll_mean_12', 'pm10_lag_24', 'so2_roll_mean_24',
        'o3_roll_mean_24', 'pm10_lag_12', 'no2_lag_24', 'pm10_roll_mean_24',
        'pm2_5_lag_24', 'aqi_roll_std_24', 'humidity_roll_mean_24', 'dayofweek',
        'pm2_5_roll_mean_24', 'pm2_5_roll_std_24', 'o3_roll_std_24', 'so2_roll_std_24',
        'co_lag_24', 'no2_roll_std_24', 'co_roll_mean_24', 'so2_lag_24',
        'aqi_t_plus_1', 'aqi_t_plus_6', 'aqi_t_plus_12', 'aqi_t_plus_24'
    ]
    
    df_features = df.reindex(columns=feature_cols, fill_value=0)
    print(f"✅ Computed realtime features: {df_features.shape}")
    return df_features


# ------------------------- Push Features to Hopsworks ------------------------- #
def push_to_hopsworks(
    df: pd.DataFrame,
    fs,
    fg_name: str,
    version: int,
    primary_key: list,
    event_time: str,
    description: str,
    online_enabled: bool = False
):
    """
    Create or update a Hopsworks Feature Group and insert data.
    """
    df[event_time] = pd.to_datetime(df[event_time], errors="coerce", utc=True)
    df = df.dropna(subset=[event_time])
    df = df.reset_index(drop=True)
    
    fg = fs.get_or_create_feature_group(
        name=fg_name,
        version=version,
        primary_key=primary_key,
        event_time=event_time,
        description=description,
        online_enabled=online_enabled
    )
    
    fg.insert(df, write_options={"wait_for_job": True})
    print(f"✅ Successfully pushed {len(df)} records to feature group: '{fg_name}' (v{version})")


# ------------------------- Main Script ------------------------- #
if __name__ == "__main__":
    # Connect to Hopsworks
    project, fs = connect_hopsworks()
    
    # Fetch raw observations
    raw_fg = fs.get_feature_group("raw_observations", version=2)
    df_raw = raw_fg.read()
    df_raw = df_raw.sort_values("datetime").reset_index(drop=True)
    print(f"✅ Fetched raw observations: {df_raw.shape}")
    
    # Compute features
    df_features = compute_features(df_raw)
    
    # Push computed features to Hopsworks
    push_to_hopsworks(
        df=df_features,
        fs=fs,
        fg_name="computed_features_realtime",
        version=1,
        primary_key=["datetime"],
        event_time="datetime",
        description="Real-time computed features for AQI model (35-feature format, Karachi)",
        online_enabled=False
    )
