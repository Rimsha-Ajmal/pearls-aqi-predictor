# -*- coding: utf-8 -*-
"""
fetch_realtime_data.py

Fetch current AQI + weather readings from APIs, validate them,
and insert into 'raw_observations' Feature Group in Hopsworks.
"""

import os
import requests
import pandas as pd
import datetime as dt
import hopsworks

# -----------------------------
# CONFIGURATION
# -----------------------------
AQI_API_URL = "https://api.waqi.info/feed/here/?token=" + os.environ.get("WAQI_TOKEN", "")
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"
LAT, LON = 24.8607, 67.0011  # Karachi (change if needed)


# -----------------------------
# FETCH FUNCTIONS
# -----------------------------
def fetch_aqi_data():
    resp = requests.get(AQI_API_URL)
    data = resp.json()
    if "data" not in data or not isinstance(data["data"], dict):
        raise ValueError("Invalid AQI API response")

    iaqi = data["data"].get("iaqi", {})
    aqi_value = data["data"].get("aqi", None)

    return {
        "aqi": aqi_value,
        "pm2_5": iaqi.get("pm25", {}).get("v"),
        "pm10": iaqi.get("pm10", {}).get("v"),
        "co": iaqi.get("co", {}).get("v"),
        "no2": iaqi.get("no2", {}).get("v"),
        "so2": iaqi.get("so2", {}).get("v"),
        "o3": iaqi.get("o3", {}).get("v"),
    }


def fetch_weather_data():
    params = {
        "latitude": LAT,
        "longitude": LON,
        "current_weather": True,
    }
    resp = requests.get(WEATHER_API_URL, params=params)
    data = resp.json()
    current = data.get("current_weather", {})
    return {
        "temp": current.get("temperature"),
        "humidity": current.get("relativehumidity_2m"),
        "wind_speed": current.get("windspeed"),
    }


# -----------------------------
# MAIN
# -----------------------------
def main():
    print("🚀 Fetching real-time data...")

    # 1️⃣ Fetch data
    aqi_data = fetch_aqi_data()
    weather_data = fetch_weather_data()

    combined = {**aqi_data, **weather_data}
    combined["datetime"] = dt.datetime.now(dt.timezone.utc)
    combined["datetime_str"] = combined["datetime"].strftime("%Y-%m-%d %H:%M:%S")

    df = pd.DataFrame([combined])
    print(df)

    # 2️⃣ Save backup CSV
    df.to_csv("realtime_aqi_weather.csv", index=False)
    print("✅ Real-time data saved to realtime_aqi_weather.csv")

    # 3️⃣ Connect to Hopsworks
    project = hopsworks.login(api_key_value=os.environ.get("HOPSWORKS_API_KEY"))
    fs = project.get_feature_store()
    fg = fs.get_feature_group("raw_observations", version=2)

    # 4️⃣ 🩹 Fix datatypes for Hopsworks schema
    df["datetime"] = pd.to_datetime(df["datetime"])        # timestamp
    df["humidity"] = pd.to_numeric(df["humidity"], errors="coerce").fillna(0).astype(int)
    df["aqi"] = pd.to_numeric(df["aqi"], errors="coerce").fillna(0).astype(int)

    # 5️⃣ Insert into feature group
    try:
        fg.insert(df)
        print("✅ Real-time data inserted into raw_observations successfully!")
    except Exception as e:
        print(f"⚠️ Could not insert into Hopsworks: {e}")

    print("✅ Raw data fetch completed.")


if __name__ == "__main__":
    main()
