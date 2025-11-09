import streamlit as st
from utils.hops import connect_hopsworks
import os

# ---------------------- PAGE CONFIG ---------------------- #
st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon="🌫️",
    layout="wide"
)

# ---------------------- SIDEBAR ---------------------- #
st.sidebar.title("🌫️ Pearls AQI Predictor")
st.sidebar.caption("Real-time AQI tracking & ML forecasting")

# st.write("ENV OK:", os.getenv("PROJECT_NAME"))

# Hopsworks Connect
if st.sidebar.button("🔌 Connect to Hopsworks"):
    with st.spinner("Connecting to Hopsworks..."):
        connect_hopsworks()
    st.sidebar.success("✅ Connected!")

st.sidebar.markdown("---")
st.sidebar.subheader("📍 Navigation")

# Page Selector
page = st.sidebar.radio(
    "Choose a page",
    [
        "🏠 Home",
        "⏱️ Real-time AQI",
        "🤖 Model Training",
        "📈 Model Insights",
        "🌫️ Predict AQI",
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("Use the menu above to navigate")

# ---------------------- PAGE ROUTING ---------------------- #
if page == "🏠 Home":
    st.title("🌍 Air Quality Monitoring & Prediction System")

    st.write("""
    Welcome to the **Pearls AQI Predictor** —  
    a smart AI-powered system delivering real-time air quality insights and advanced AQI forecasting.

    ### 🚀 Features
    - 📡 Real-time Weather & AQI Monitoring  
    - 🤖 ML-powered AQI Prediction  
    - 📊 Feature Analytics & Insights  
    - 🧠 Model Training & Explainability  
    """)

    st.image(
        "https://images.unsplash.com/photo-1504610926078-a1611febcad3?w=1400",
        caption="Smarter Air Quality Awareness for a Healthier City",
        # width="auto"
    )

elif page == "📈 Model Insights":
    import pages.model_insights as pg
    pg.app()

elif page == "🤖 Model Training":
    import pages.model_training as pg
    pg.app()

elif page == "🌫️ Predict AQI":
    import pages.predict_AQI as pg
    pg.app()

elif page == "⏱️ Real-time AQI":
    import pages.realtime_data as pg
    pg.app()
