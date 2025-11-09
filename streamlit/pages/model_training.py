import streamlit as st
import pandas as pd
import os
import json
from dotenv import load_dotenv
from utils.hops import connect_hopsworks

load_dotenv()  # ensure env vars available if needed

# ---------- Paths ----------
ALL_METRICS_PATH = "models/all_models_metrics.json"
LAST_LOG_PATH = "models/last_training_log.txt"
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ---------- Helper Functions ----------
def load_all_metrics():
    if os.path.exists(ALL_METRICS_PATH):
        with open(ALL_METRICS_PATH, "r") as f:
            return json.load(f)
    return []

def show_local_fallback():
    if os.path.exists(LAST_LOG_PATH):
        with open(LAST_LOG_PATH, "r", encoding="utf-8") as f:
            logs = f.read()
        st.warning("⚠️ Showing locally saved training results (offline mode).")
        st.code(logs, language="bash")

# ---------- Streamlit App ----------
def app():
    # ----- Page Header -----
    st.markdown("<h1 style='text-align: center; color: #4B0082;'>🤖 Train AQI Prediction Models</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #555;'>Train, compare, and select the best AQI prediction model</h4>", unsafe_allow_html=True)
    st.markdown("---")

    # ----- Load latest features -----
    st.markdown("<h2 style='color:#4B0082;'>📋 Computed Features Snapshot</h2>", unsafe_allow_html=True)
    local_cache = "data/recent_snapshot.csv"
    try:
        project, fs = connect_hopsworks()
        fg = fs.get_feature_group("computed_features_historical", version=1)

        df_features = fg.read()
        if df_features is None or df_features.empty:
            raise ValueError("Feature group returned empty or None.")

        df_features = df_features.sort_values("datetime").tail(10)
        df_features.to_csv(local_cache, index=False)
        st.success("✅ Loaded recent observation snapshot from Hopsworks.")

    except Exception as e:
        st.warning(f"⚠️ Could not fetch data from Hopsworks: {e}")

        if os.path.exists(local_cache):
            df_features = pd.read_csv(local_cache)
            st.info("📁 Loaded cached snapshot from previous run.")
        else:
            st.error("❌ No cached data found. Please connect once to Hopsworks to cache it.")
            df_features = None  # fallback

    if df_features is not None:
        st.dataframe(df_features, width='stretch')

    st.markdown("---")
    st.markdown("<h2 style='color:#4B0082;'>📊 Latest Training Results</h2>", unsafe_allow_html=True)

    # ----- Load all metrics -----
    all_metrics = load_all_metrics()
    if all_metrics:
        df_metrics = pd.DataFrame(all_metrics)
        df_metrics = df_metrics.sort_values("R2", ascending=False).reset_index(drop=True)

        # Display metrics in card-style layout
        st.markdown("### 🚀 Model Comparison:")
        col_count = 3  # display metrics in 3-column cards
        rows = (len(df_metrics) + col_count - 1) // col_count

        for i in range(rows):
            cols = st.columns(col_count, gap="medium")
            for j in range(col_count):
                idx = i * col_count + j
                if idx < len(df_metrics):
                    model = df_metrics.iloc[idx]
                    card_style = """
                        background-color: #f0f4f8; 
                        padding: 20px; 
                        border-radius: 10px;
                        text-align: center;
                        margin-bottom: 10px;
                    """
                    cols[j].markdown(f"""
                        <div style="{card_style}">
                            <h4>🏷️ {model['Model']}</h4>
                            <p>📈 MAE: {model['MAE']:.3f}</p>
                            <p>📉 RMSE: {model['RMSE']:.3f}</p>
                            <p>💯 R²: {model['R2']:.3f}</p>
                        </div>
                    """, unsafe_allow_html=True)

        best_model = df_metrics.iloc[0]
        st.markdown("---")
        st.markdown(
            f"🏆 **Best Model:** `{best_model['Model']}`\n\n"
            f"📈 **Metrics** → MAE: `{best_model['MAE']:.3f}`, "
            f"RMSE: `{best_model['RMSE']:.3f}`, "
            f"R²: `{best_model['R2']:.3f}`"
        )

    else:
        st.warning("⚠️ No metrics found. Run training first or load offline results.")
        show_local_fallback()


if __name__ == "__main__":
    app()
