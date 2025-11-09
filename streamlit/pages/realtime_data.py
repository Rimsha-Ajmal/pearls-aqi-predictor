import streamlit as st
import plotly.express as px
import pandas as pd
from utils.hops import connect_hopsworks

# ------------------------- Load Recent Data -------------------------
@st.cache_data(show_spinner="📡 Fetching recent AQI data...")
def load_recent_data(limit=50):
    """Fetch the most recent records from raw_observations."""
    project, fs = connect_hopsworks()
    fg = fs.get_feature_group(name="raw_observations", version=2)
    df = fg.read()
    
    # Ensure datetime is parsed correctly
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    
    # Sort descending by datetime to get latest first
    df.sort_values("datetime", ascending=False, inplace=True)
    
    # Reset index
    df.reset_index(drop=True, inplace=True)
    
    return df.head(limit)

# ------------------------- Streamlit App -------------------------
def app():
    st.title("📡 Real-Time Hourly AQI & Weather Data")

    try:
        df = load_recent_data()
    except Exception as e:
        st.error("❌ Could not load data. Check your Hopsworks connection or internet.")
        st.exception(e)
        return

    st.success(f"✅ Live data loaded — showing latest {len(df)} records")

    # ---------------- Latest Observation ----------------
    st.subheader("📍 Latest Observation")
    latest_record = df.iloc[0]
    st.dataframe(latest_record.to_frame().T, width='stretch')

    # ---------------- Recent Observations ----------------
    st.subheader("📑 Recent Observations")
    st.dataframe(df, width='stretch')

    # ---------------- AQI Trend (Last 48 Hours) ----------------
    st.subheader("📈 AQI Trend (Recent Hours)")
    if len(df) > 1:
        # Sort ascending for a proper line chart (oldest to newest)
        fig = px.line(
            df.sort_values("datetime").head(48),
            x="datetime",
            y="aqi",
            markers=True,
            title="Hourly AQI Levels (Latest 48 Records)"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("ℹ️ Not enough data to show AQI trend.")

    # ---------------- Latest Weather Snapshot ----------------
    weather_cols = [c for c in df.columns if any(w in c.lower() for w in ["temp", "humidity", "wind", "pressure"])]
    if weather_cols:
        st.subheader("🌦 Weather Conditions (Latest)")
        st.dataframe(df.loc[0, weather_cols].to_frame().T, width='stretch')
    else:
        st.info("ℹ️ No weather data available in this dataset.")
