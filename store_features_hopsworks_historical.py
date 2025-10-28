"""
push_to_hopsworks.py
-----------------------------------
Script to upload engineered features to Hopsworks Feature Store.
"""

import os
import hopsworks
import pandas as pd


# -----------------------------
# 1️⃣ Load Hopsworks API Key
# -----------------------------
# Set your API key as an environment variable before running:
#   export HOPSWORKS_API_KEY="your_api_key_here"
# or on Windows (PowerShell):
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
# 4️⃣ Load Historical Feature Data
# -----------------------------
historical_path = "model_features_v1.csv"

if not os.path.exists(historical_path):
    raise FileNotFoundError(f"❌ File not found: {historical_path}")

df_features = pd.read_csv(historical_path)
print(f"✅ Loaded {len(df_features)} historical records from {historical_path}")
print("📊 Columns:", len(df_features.columns))


# -----------------------------
# 5️⃣ Push Data to Hopsworks
# -----------------------------
push_to_hopsworks(
    df=df_features,
    fg_name="model_features",
    version=1,
    primary_key=["datetime"],
    event_time="datetime",
    description="Engineered AQI + Weather features (Karachi, hourly)",
    online_enabled=False
)


# -----------------------------
# 6️⃣ (Optional) Delete a Feature Group
# -----------------------------
# Uncomment if you want to delete a feature group version.
"""
fg = fs.get_feature_group(name="aqi_features", version=1)
fg.delete()
print("✅ Deleted feature group 'aqi_features'")
"""

print("🎯 Script completed successfully!")
