import os
import requests
import pandas as pd
from datetime import datetime, timezone
import hopsworks  # ✅ new import

# --- API Key ---
API_KEY = os.getenv("OPENWEATHER_API_KEY")

LAT, LON = 24.8607, 67.0011  # Karachi

def get_realtime_data():
    try:
        # Weather API
        weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
        weather = requests.get(weather_url, timeout=10).json()

        # Air Pollution API
        pollution_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"
        pollution = requests.get(pollution_url, timeout=10).json()

        if "main" not in weather or "list" not in pollution:
            raise ValueError("Invalid API response format")

        temp = weather["main"]["temp"]
        humidity = weather["main"]["humidity"]
        wind_speed = weather["wind"]["speed"]

        air = pollution["list"][0]
        aqi = air["main"]["aqi"]
        comp = air["components"]

        data = {
            "datetime": datetime.now(timezone.utc),
            "datetime_str": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "city": "Karachi",
            "temp": temp,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "aqi": aqi,
            "pm2_5": comp.get("pm2_5"),
            "pm10": comp.get("pm10"),
            "co": comp.get("co"),
            "no2": comp.get("no2"),
            "so2": comp.get("so2"),
            "o3": comp.get("o3"),
        }

        return data

    except Exception as e:
        print(f"⚠️ Error fetching data: {e}")
        return None


data = get_realtime_data()
if data:
    df = pd.DataFrame([data])
    print(df)

    # ✅ Save locally (optional)
    filename = "realtime_aqi_weather.csv"
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
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
