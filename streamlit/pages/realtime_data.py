import streamlit as st
import plotly.express as px
from utils.hops import connect_hopsworks
import pandas as pd


@st.cache_data(show_spinner="📡 Fetching recent AQI data...")
def load_recent_data(limit=50):
    """Fetch only the most recent records from raw_observations."""
    project, fs = connect_hopsworks()
    fg = fs.get_feature_group(name="raw_observations", version=2)
    df = fg.read()
    df.sort_values("datetime", ascending=False, inplace=True)
    return df.head(limit)

def app():
    st.title("📡 Real-Time Hourly AQI & Weather Data")

    try:
        df = load_recent_data()
    except Exception as e:
        st.error("❌ Could not load data. Check your Hopsworks connection or internet.")
        st.exception(e)
        return

    st.success(f"✅ Live data loaded — showing latest {len(df)} records")

    # 📍 Latest Observation
    st.subheader("📍 Latest Observation")
    st.dataframe(df.head(1), width='stretch')

    # 📈 AQI Trend
    st.subheader("📈 AQI Trend (Recent Hours)")
    fig = px.line(
        df.sort_values("datetime").tail(48),
        x="datetime",
        y="aqi",
        markers=True,
        title="Hourly AQI Levels"
    )
    st.plotly_chart(fig, width='stretch')

    # 🌦 Weather Snapshot
    weather_cols = [c for c in df.columns if any(w in c.lower() for w in ["temp", "humidity", "wind", "pressure"])]
    if weather_cols:
        st.subheader("🌦 Weather Conditions (Latest)")
        st.dataframe(df[weather_cols].head(1), width='stretch')
