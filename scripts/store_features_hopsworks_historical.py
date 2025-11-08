# store_features_hopsworks_historical.py

import os
import pandas as pd
import hopsworks

# ---------------- LOAD API KEY ----------------
# Expecting the Hopsworks API key to be stored in an environment variable for security.
# You can set it in your terminal or .env file as:
# export HOPSWORKS_API_KEY="your_api_key_here"

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

if not HOPSWORKS_API_KEY:
    raise ValueError("❌ HOPSWORKS_API_KEY environment variable not set!")

print("🔐 API key loaded successfully.")

# ---------------- CONNECT TO HOPSWORKS ----------------
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()
print("✅ Connected to Hopsworks project successfully!")

# ---------------- DEFINE FUNCTION ----------------
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

    # Insert data into Hopsworks
    fg.insert(df, write_options={"wait_for_job": True})
    print(f"✅ Successfully pushed {len(df)} records to feature group: '{fg_name}' (v{version})")


# ---------------- MAIN EXECUTION ----------------
if __name__ == "__main__":
    historical_path = "computed_features_historical_selected_cleaned.csv"

    df_features = pd.read_csv(historical_path)
    print(f"✅ Loaded {len(df_features)} historical records from {historical_path}")
    print(f"📊 Columns: {len(df_features.columns)}")

    # Push computed features to Hopsworks
    push_to_hopsworks(
        df=df_features,
        fg_name="computed_features_historical_v3",
        version=1,
        primary_key=["datetime"],
        event_time="datetime",
        description="Computed Historical AQI + Weather features (Karachi, hourly)",
        online_enabled=False
    )

    # Push raw historical data (optional reuse)
    push_to_hopsworks(
        df=df_features,
        fg_name="historical_data",
        version=1,
        primary_key=["datetime"],
        event_time="datetime",
        description="Historical AQI + Weather Data (Karachi, hourly)",
        online_enabled=False
    )
