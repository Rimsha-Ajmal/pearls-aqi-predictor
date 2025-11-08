import streamlit as st
import pandas as pd
import plotly.express as px
from utils.hops import connect_hopsworks
import os
import json

# ---------- Paths ----------
METRICS_PATH = "models/last_metrics.json"
os.makedirs("models", exist_ok=True)

# ---------- Helper Functions ----------
def load_model_metrics(from_hopsworks=False):
    """Load metrics & feature importance from Hopsworks or local cache."""
    metrics, feature_importance = {}, {}

    if from_hopsworks:
        try:
            project, _ = connect_hopsworks()
            mr = project.get_model_registry()
            models = mr.get_models()

            if not models:
                raise Exception("No models found in registry.")

            latest = sorted(models, key=lambda m: m.version, reverse=True)[0]
            details = latest.get_model()

            metrics = {
                "Model Name": latest.name,
                "Version": latest.version,
                **details.get("metrics", {})
            }
            feature_importance = details.get("feature_importance", {})

            # ✅ Cache locally for offline access
            with open(METRICS_PATH, "w") as f:
                json.dump(metrics, f, indent=2)
            with open("models/last_feature_importance.json", "w") as f:
                json.dump(feature_importance, f, indent=2)

            st.success(f"✅ Loaded latest model: {latest.name} (v{latest.version})")
            return metrics, feature_importance

        except Exception as e:
            st.warning(f"⚠️ Could not fetch from Hopsworks: {e}")

    # ✅ Offline fallback
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
        if os.path.exists("models/last_feature_importance.json"):
            with open("models/last_feature_importance.json", "r") as f:
                feature_importance = json.load(f)
        st.warning("⚠️ Showing locally saved metrics (offline mode)")
        return metrics, feature_importance

    st.error("❌ No model information found.")
    return None, None


# ---------- Streamlit App ----------
def app():
    st.title("🧠 Model Insights & Performance")

    # --- Control buttons ---
    col1, col2 = st.columns([0.5, 0.5])
    with col1:
        if st.button("🔄 Refresh from Hopsworks"):
            with st.spinner("Fetching model details..."):
                metrics, feature_importance = load_model_metrics(from_hopsworks=True)
                if metrics:
                    st.rerun()
                else:
                    st.error("❌ Could not refresh model info.")
                    return
    with col2:
        st.caption("Local cache loads instantly when offline.")

    # --- Load cached/online data ---
    metrics, feature_importance = load_model_metrics()

    if metrics:
        # 📊 Metrics
        st.subheader("📊 Model Performance Metrics")
        st.table(pd.DataFrame([metrics]))

        # 📈 Feature Importance
        if feature_importance:
            st.subheader("📈 Feature Importance")
            fi_df = pd.DataFrame(list(feature_importance.items()), columns=["Feature", "Importance"])
            fi_df.sort_values("Importance", ascending=False, inplace=True)
            fig = px.bar(fi_df, x="Feature", y="Importance", title="Feature Importance")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("ℹ️ Feature importance not available for this model.")
