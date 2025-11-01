"""
push_realtime_to_hopsworks.py
------------------------------------------------
Script to upload real-time AQI + Weather features
to a Hopsworks Feature Store.
"""

import os
import pandas as pd
import hopsworks


# -----------------------------
# 1️⃣ Load Hopsworks API Key
# -----------------------------
# Before running:
#   export HOPSWORKS_API_KEY="your_api_key_here"
# or (Windows PowerShell):
#   setx HOPSWORKS_API_KEY "your_api_key_here"

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

if not HOPSWORKS_API_KEY:
    raise ValueError("❌ HOPSWORKS_API_KEY environment variable not found. Please set it first.")

print("🔑 API key loaded successfully!")


# -----------------------------
# 2️⃣ Connect to Hopsworks Project
# -----------------------------
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()
print("✅ Connected to Hopsworks project successfully!")


# -----------------------------
# 3️⃣ Define Function to Push Data
# -----------------------------
def push_to_hopsworks(
    df: pd.DataFrame,
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
    # Ensure datetime formatting
    df[event_time] = pd.to_datetime(df[event_time], errors="coerce", utc=True)
    df = df.dropna(subset=[event_time])
    df = df.reset_index(drop=True)

    # Create or get feature group
    fg = fs.get_or_create_feature_group(
        name=fg_name,
        version=version,
        primary_key=primary_key,
        event_time=event_time,
        description=description,
        online_enabled=online_enabled
    )

    # Insert data
    fg.insert(df, write_options={"wait_for_job": True})
    print(f"✅ Successfully pushed {len(df)} records to feature group: '{fg_name}' (v{version})")


# -----------------------------
# 4️⃣ Load Real-Time Feature Data
# -----------------------------
realtime_path = "realtime_aqi_weather.csv"

if not os.path.exists(realtime_path):
    raise FileNotFoundError(f"❌ File not found: {realtime_path}")

df_features = pd.read_csv(realtime_path)
print(f"✅ Loaded {len(df_features)} real-time records from {realtime_path}")
print("📊 Columns:", len(df_features.columns))


# -----------------------------
# 5️⃣ Data Preprocessing
# -----------------------------
if "pm25" in df_features.columns:
    df_features.rename(columns={"pm25": "pm2_5"}, inplace=True)

df_features["datetime"] = pd.to_datetime(df_features["datetime"], utc=True)
df_features["datetime_str"] = df_features["datetime"].astype(str)  # string key for uniqueness


# -----------------------------
# 6️⃣ Push Data to Hopsworks
# -----------------------------
push_to_hopsworks(
    df=df_features,
    fg_name="raw_observations",
    version=2,
    primary_key=["datetime_str"],
    event_time="datetime",
    description="Real-time AQI + Weather features (Karachi, hourly)",
    online_enabled=True
)

print("🎯 Script completed successfully!")
