import streamlit as st
from utils.hops import connect_hopsworks
import os

# ---------------------- PAGE CONFIG ---------------------- #
st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------- SIDEBAR ---------------------- #
st.sidebar.markdown("<h2 style='text-align: center; color: #4B0082;'>📊 DASHBOARD</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Hopsworks Connect Button
if st.sidebar.button("🔌 Connect to Hopsworks"):
    st.info("Connecting to Hopsworks...")
    connect_hopsworks()
    st.success("✅ Connected!")

# Sidebar Navigation
st.sidebar.markdown("<h4 style='color:#4B0082; margin-top:10px;'>📍 Navigation</h4>", unsafe_allow_html=True)

# Using a list and loop to reduce extra spacing
pages = ["🏠 Home", "📈 Model Insights", "🤖 Model Training", "🌫️ Predict AQI", "⏱️ Real-time AQI"]
page = st.sidebar.radio("", pages, index=0, key="sidebar_nav")

st.sidebar.markdown("---")
st.sidebar.info("💡 Use the menu above to navigate through pages")

# ---------------------- HOME PAGE ---------------------- #
if page == "🏠 Home":
    # Page Header
    st.markdown("<h1 style='text-align: center; color: #4B0082;'>🌍 Air Quality Monitoring & Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #555;'>Smart AI-powered system delivering real-time AQI insights</h3>", unsafe_allow_html=True)
    st.markdown("---")

    # Features as modern cards with background
    st.markdown("<h2 style='color:#4B0082;'>🚀 Features</h2>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4, gap="medium")

    card_style = """
    background-color: #f0f4f8; 
    padding: 20px; 
    border-radius: 10px;
    text-align: center;
    """

    with col1:
        st.markdown(f"<div style='{card_style}'><h3>📡 Real-time Monitoring</h3><p>Track live weather & AQI data with smart visualizations.</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div style='{card_style}'><h3>🤖 ML Prediction</h3><p>Predict AQI for the next 3 days using AI-powered models.</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div style='{card_style}'><h3>📊 Analytics & Insights</h3><p>Understand key features affecting air quality.</p></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div style='{card_style}'><h3>🧠 Explainability</h3><p>Train models & analyze feature importance with SHAP.</p></div>", unsafe_allow_html=True)

    st.markdown("---")

    # Image Section with rounded corners and container width
    st.image(
        "assets/air-quality-index.webp",
        caption="Smarter Air Quality Awareness for a Healthier City",
        width='stretch'
    )

# ---------------------- OTHER PAGES ---------------------- #
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


