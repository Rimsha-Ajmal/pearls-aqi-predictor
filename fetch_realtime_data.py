import os
import requests
import pandas as pd
from datetime import datetime, timezone
import numpy as np
import hopsworks  # ✅ Hopsworks integration

# --- API Key ---
API_KEY = os.getenv("OPENWEATHER_API_KEY")
LAT, LON = 24.8607, 67.0011  # Karachi


def get_realtime_data():
    try:
        # 🌤️ Weather API
        weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
        weather = requests.get(weather_url, timeout=10).json()

        # 💨 Air Pollution API
        pollution_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"
        pollution = requests.get(pollution_url, timeout=10).json()

        if "main" not in weather or "list" not in pollution:
            raise ValueError("Invalid API response format")

        # Extract weather data safely
        temp = float(weather["main"].get("temp", np.nan))
        humidity = float(weather["main"].get("humidity", np.nan))
        wind_speed = float(weather.get("wind", {}).get("speed", np.nan))

        air = pollution["list"][0]
        aqi = float(air["main"].get("aqi", np.nan))
        comp = air.get("components", {})

        # Prepare data dictionary
        now_utc = datetime.now(timezone.utc)
        data = {
            "datetime": now_utc.isoformat(),  # ✅ ISO 8601 format (YAML + Hopsworks safe)
            "datetime_str": now_utc.strftime("%Y-%m-%d %H:%M:%S"),
            "city": "Karachi",
            "temp": temp,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "aqi": aqi,
            "pm2_5": float(comp.get("pm2_5", np.nan)),
            "pm10": float(comp.get("pm10", np.nan)),
            "co": float(comp.get("co", np.nan)),
            "no2": float(comp.get("no2", np.nan)),
            "so2": float(comp.get("so2", np.nan)),
            "o3": float(comp.get("o3", np.nan)),
        }

        # Replace invalid numeric values with None (safe for YAML & Hopsworks)
        for key, value in data.items():
            if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                data[key] = None

        return data

    except Exception as e:
        print(f"⚠️ Error fetching data: {e}")
        return None


# --- Main Execution ---
data = get_realtime_data()
if data:
    df = pd.DataFrame([data])
    print(df)

    # ✅ Save locally (optional backup)
    filename = "realtime_aqi_weather.csv"
    if os.path.exists(filename):
        df.to_csv(filename, mode="a", header=False, index=False)
    else:
        df.to_csv(filename, index=False)
    print(f"✅ Real-time data saved to {filename}")

    # ✅ Push to Hopsworks
    try:
        project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
        fs = project.get_feature_store()

        fg = fs.get_feature_group(name="raw_observations", version=2)
        fg.insert(df)
        print("✅ Data inserted into Hopsworks feature group 'raw_observations'")
    except Exception as e:
        print(f"⚠️ Could not insert into Hopsworks: {e}")

else:
    print("❌ No data fetched.")