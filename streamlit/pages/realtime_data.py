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
    # ----- Page Header -----
    st.markdown("<h1 style='text-align: center; color: #4B0082;'>📡 Real-Time Hourly AQI & Weather Data</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #555;'>Monitor live air quality and weather observations</h4>", unsafe_allow_html=True)
    st.markdown("---")

    try:
        df = load_recent_data()
    except Exception as e:
        st.error("❌ Could not load data. Check your Hopsworks connection or internet.")
        st.exception(e)
        return

    st.success(f"✅ Live data loaded — showing latest {len(df)} records")

    # ---------------- Latest Observation ----------------
    st.markdown("<h2 style='color:#4B0082;'>📍 Latest Observation</h2>", unsafe_allow_html=True)
    latest_record = df.iloc[0]
    card_style = """
        background-color: #f0f4f8; 
        padding: 15px; 
        border-radius: 10px;
        text-align: center;
        margin-bottom: 10px;
    """
    st.markdown(f"<div style='{card_style}'>{latest_record.to_frame().T.to_html(index=False)}</div>", unsafe_allow_html=True)

    # ---------------- Recent Observations ----------------
    st.markdown("<h2 style='color:#4B0082;'>📑 Recent Observations</h2>", unsafe_allow_html=True)
    st.dataframe(df, width='stretch')

    # ---------------- AQI Trend (Last 48 Hours) ----------------
    st.markdown("<h2 style='color:#4B0082;'>📈 AQI Trend (Recent Hours)</h2>", unsafe_allow_html=True)
    if len(df) > 1:
        # Sort ascending for a proper line chart (oldest to newest)
        fig = px.line(
            df.sort_values("datetime").head(48),
            x="datetime",
            y="aqi",
            markers=True,
            title="Hourly AQI Levels (Latest 48 Records)",
            template="plotly_white"
        )
        fig.update_layout(title_font_color="#4B0082", xaxis_title="Time", yaxis_title="AQI")
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("ℹ️ Not enough data to show AQI trend.")

    # ---------------- Latest Weather Snapshot ----------------
    weather_cols = [c for c in df.columns if any(w in c.lower() for w in ["temp", "humidity", "wind", "pressure"])]
    st.markdown("<h2 style='color:#4B0082;'>🌦 Latest Weather Conditions</h2>", unsafe_allow_html=True)
    if weather_cols:
        latest_weather = df.loc[0, weather_cols].to_frame().T
        st.markdown(f"<div style='{card_style}'>{latest_weather.to_html(index=False)}</div>", unsafe_allow_html=True)
    else:
        st.info("ℹ️ No weather data available in this dataset.")
