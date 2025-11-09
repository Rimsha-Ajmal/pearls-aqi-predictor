import streamlit as st
import pandas as pd
import plotly.express as px
import shap
import joblib
import numpy as np
from utils.hops import connect_hopsworks
import os
import json
import shutil

# ---------- Paths ----------
METRICS_PATH = "models/last_metrics.json"
SHAP_JSON_PATH = "models/shap_feature_importance.json"
MODEL_PATH = "models/randomForest_shap_30_model.pkl"
os.makedirs("models", exist_ok=True)

# ---------- Helper Functions ----------
def load_model_from_registry(model_name="randomForest_shap_30_model", local_path=MODEL_PATH):
    """Load model from local path or Hopsworks model registry."""
    if os.path.exists(local_path):
        st.success("✅ Loaded local model successfully.")
        return joblib.load(local_path)

    st.info("🔍 Local model not found — attempting to fetch from Hopsworks...")
    try:
        project, _ = connect_hopsworks()
        mr = project.get_model_registry()
        model_versions = mr.get_models(name=model_name)
        if not model_versions:
            st.error(f"❌ No model named '{model_name}' found in registry.")
            return None

        latest_model = model_versions[-1]
        model_dir = latest_model.download()

        found_file = None
        for root, _, files in os.walk(model_dir):
            for f in files:
                if f.endswith((".pkl", ".joblib")):
                    found_file = os.path.join(root, f)
                    break
            if found_file:
                break

        if not found_file:
            st.error("❌ No .pkl or .joblib file found inside model directory.")
            return None

        shutil.copy(found_file, local_path)
        st.success(f"✅ Downloaded and cached model version {latest_model.version} from Hopsworks!")
        return joblib.load(local_path)

    except Exception as e:
        st.error(f"❌ Could not load model from Hopsworks: {e}")
        return None

def load_model_metrics(from_hopsworks=False):
    """Load metrics from Hopsworks or local cache."""
    metrics = {}
    if from_hopsworks:
        try:
            project, _ = connect_hopsworks()
            mr = project.get_model_registry()
            models = mr.get_models(name="randomForest_shap_30_model")
            if not models:
                raise Exception("No models found in registry.")

            latest = sorted(models, key=lambda m: m.version, reverse=True)[0]
            details = latest.get_model()

            metrics = {
                "Model Name": latest.name,
                "Version": latest.version,
                **details.get("metrics", {})
            }

            with open(METRICS_PATH, "w") as f:
                json.dump(metrics, f, indent=2)

            st.success(f"✅ Loaded latest model metrics: {latest.name} (v{latest.version})")
            return metrics

        except Exception as e:
            st.warning(f"⚠️ Could not fetch from Hopsworks: {e}")

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
        return metrics

    st.error("❌ No model information found.")
    return None

def load_shap_json(path=SHAP_JSON_PATH):
    """Load SHAP feature importance from JSON if exists."""
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

def compute_shap_from_feature_group(model_path=MODEL_PATH, fg_name="computed_features_historical", fg_version=1):
    """Fetch Feature Group, download model if needed, compute SHAP values, save JSON."""
    try:
        project, fs = connect_hopsworks()
        fg = fs.get_feature_group(name=fg_name, version=fg_version)
        df = fg.read()
        st.info(f"✅ Loaded {df.shape[0]} rows from Feature Group '{fg_name}'")
    except Exception as e:
        st.error(f"❌ Could not read Feature Group '{fg_name}': {e}")
        return None

    EXCLUDE_COLS = ['datetime']
    X = df.drop(columns=[c for c in EXCLUDE_COLS if c in df.columns], errors='ignore')
    X_sample = X.sample(min(500, len(X)), random_state=42)

    model = load_model_from_registry(local_path=model_path)
    if model is None:
        st.error("❌ Cannot compute SHAP without a model.")
        return None

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    shap_df = pd.DataFrame({
        "Feature": X_sample.columns,
        "Mean |SHAP|": np.abs(shap_values).mean(axis=0)
    }).sort_values("Mean |SHAP|", ascending=False)

    shap_dict = dict(zip(shap_df["Feature"], shap_df["Mean |SHAP|"]))
    with open(SHAP_JSON_PATH, "w") as f:
        json.dump(shap_dict, f, indent=2)

    return shap_df

# ---------- Streamlit App ----------
def app():
    st.title("🧠 Model Insights & Performance")

    # --- Refresh button ---
    col1, col2 = st.columns([0.5, 0.5])
    with col1:
        if st.button("🔄 Refresh from Hopsworks"):
            with st.spinner("Fetching model details..."):
                metrics = load_model_metrics(from_hopsworks=True)
                if metrics:
                    st.rerun()
                else:
                    st.error("❌ Could not refresh model info.")
                    return
    with col2:
        st.caption("Local cache loads instantly when offline.")

    # --- Load metrics ---
    metrics = load_model_metrics()
    if metrics:
        st.subheader("📊 Model Performance Metrics")
        st.table(pd.DataFrame([metrics]))

    # --- SHAP feature importance ---
    st.subheader("🧩 SHAP Feature Importance")
    shap_dict = load_shap_json()
    if shap_dict:
        shap_df = pd.DataFrame(list(shap_dict.items()), columns=["Feature", "Mean |SHAP|"])
        shap_df.sort_values("Mean |SHAP|", ascending=False, inplace=True)
    else:
        with st.spinner("Computing SHAP values from Feature Group (may take a while)..."):
            shap_df = compute_shap_from_feature_group()

    if shap_df is not None:
        st.table(shap_df)
        fig = px.bar(shap_df, x="Feature", y="Mean |SHAP|", title="SHAP Feature Importance")
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("ℹ️ SHAP values cannot be computed. Ensure model & feature group exist.")

# Run app
if __name__ == "__main__":
    app()
