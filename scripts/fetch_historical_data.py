# -*- coding: utf-8 -*-
"""
fetch_historical_data.py

Fetch historical weather and AQI data for Karachi using Open-Meteo and OpenWeatherMap APIs,
combine them by timestamp, and save the result as a CSV file.
"""

import os
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

# -------------------------------
# CONFIGURATION
# -------------------------------
LAT, LON = 24.8607, 67.0011  # Karachi

# Load API key from environment variable (instead of Colab userdata)
API_KEY = os.environ.get("OPENWEATHER_API_KEY")

if not API_KEY:
    raise ValueError("⚠️ API key not found. Please set the environment variable 'OPENWEATHER_API_KEY'.")
print("✅ API key loaded successfully!")


# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def to_unix(date_str):
    """Convert 'YYYY-MM-DD' to UNIX timestamp."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return int(dt.replace(tzinfo=timezone.utc).timestamp())


def fetch_weather_data(start_date, end_date):
    """
    Fetch hourly historical weather data (temperature, humidity, wind speed)
    from Open-Meteo API.
    """
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={LAT}&longitude={LON}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    )

    response = requests.get(url, timeout=15)
    data = response.json()

    if "hourly" not in data or not data["hourly"].get("time"):
        raise ValueError("⚠️ No weather data returned. Check date range or parameters.")

    weather_df = pd.DataFrame({
        "datetime": data["hourly"]["time"],
        "temp": data["hourly"]["temperature_2m"],
        "humidity": data["hourly"]["relative_humidity_2m"],
        "wind_speed": data["hourly"]["wind_speed_10m"],
    })

    weather_df["datetime"] = pd.to_datetime(weather_df["datetime"], utc=True)
    return weather_df


def fetch_aqi_data(start_date, end_date):
    """
    Fetch historical AQI + pollutant data from OpenWeatherMap API.
    """
    start_ts = to_unix(start_date)
    end_ts = to_unix(end_date)

    url = (
        f"http://api.openweathermap.org/data/2.5/air_pollution/history"
        f"?lat={LAT}&lon={LON}&start={start_ts}&end={end_ts}&appid={API_KEY}"
    )

    response = requests.get(url, timeout=15)
    data = response.json()

    if "list" not in data or not data["list"]:
        raise ValueError("⚠️ No air pollution data returned. Check date range or API key.")

    records = []
    for entry in data["list"]:
        ts = entry["dt"]
        air = entry["main"]
        comp = entry["components"]
        date_time = datetime.fromtimestamp(ts, tz=timezone.utc)

        records.append({
            "datetime": date_time,
            "aqi": air["aqi"],
            "pm2_5": comp.get("pm2_5"),
            "pm10": comp.get("pm10"),
            "co": comp.get("co"),
            "no2": comp.get("no2"),
            "so2": comp.get("so2"),
            "o3": comp.get("o3")
        })

    aqi_df = pd.DataFrame(records)
    return aqi_df


def combine_weather_aqi(start_date, end_date, filename="historical_data.csv"):
    """
    Combine weather and AQI data by datetime (nearest hour) and save to CSV.
    """
    print("🌤️ Fetching weather data...")
    weather_df = fetch_weather_data(start_date, end_date)

    print("💨 Fetching air quality data...")
    aqi_df = fetch_aqi_data(start_date, end_date)

    # Ensure both datetime columns are timezone-aware (UTC)
    weather_df["datetime"] = pd.to_datetime(weather_df["datetime"], utc=True)
    aqi_df["datetime"] = pd.to_datetime(aqi_df["datetime"], utc=True)

    # Merge based on nearest timestamp (within 1 hour)
    combined_df = pd.merge_asof(
        aqi_df.sort_values("datetime"),
        weather_df.sort_values("datetime"),
        on="datetime",
        direction="nearest",
        tolerance=pd.Timedelta("1h")
    )

    combined_df.insert(1, "city", "Karachi")
    combined_df.to_csv(filename, index=False)

    print(f"✅ Combined data saved to '{filename}' ({len(combined_df)} records)")
    return combined_df


# -------------------------------
# RUN THE SCRIPT
# -------------------------------
if __name__ == "__main__":
    # Set your desired date range
    start_date = "2025-01-01"
    end_date = "2025-10-12"

    df = combine_weather_aqi(start_date, end_date)
    print(df.head())
