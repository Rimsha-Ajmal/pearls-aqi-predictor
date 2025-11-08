import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta
import plotly.express as px
import os
from utils.hops import connect_hopsworks


# ------------------------- MODEL LOADER ------------------------- #
@st.cache_resource
def load_model():
    """
    Load the AQI prediction model from local storage or Hopsworks model registry.
    """
    path = "models/AQI_RandomForest.pkl"
    os.makedirs("models", exist_ok=True)

    # ✅ If local model exists, load directly
    if os.path.exists(path):
        st.success("✅ Loaded local model successfully.")
        return joblib.load(path)

    st.info("🔍 Local model not found — attempting to fetch from Hopsworks...")
    try:
        project, _ = connect_hopsworks()
        mr = project.get_model_registry()

        # ✅ Fetch latest version automatically
        model_versions = mr.get_models(name="AQI_RandomForest")
        if not model_versions:
            raise FileNotFoundError("No model named 'AQI_RandomForest' found in registry.")

        latest_model = model_versions[-1]  # latest version
        model_dir = latest_model.download()

        # ✅ Find .pkl or .joblib file
        found_file = None
        for root, _, files in os.walk(model_dir):
            for f in files:
                if f.endswith((".pkl", ".joblib")):
                    found_file = os.path.join(root, f)
                    break
            if found_file:
                break

        if not found_file:
            raise FileNotFoundError("No .pkl or .joblib file found inside model directory.")

        os.replace(found_file, path)
        st.success(f"✅ Downloaded and cached model version {latest_model.version} from Hopsworks!")
        return joblib.load(path)

    except Exception as e:
        st.error(f"❌ Could not load model from Hopsworks: {e}")
        st.stop()


# ------------------------- AQI ALERT HELPER ------------------------- #
def aqi_alert(aqi_class):
    mapping = {
        1: ("Good", "✅ Green"),
        2: ("Moderate", "🟡 Yellow"),
        3: ("Sensitive Groups", "🟠 Orange"),
        4: ("Unhealthy", "🔴 Red"),
        5: ("Very Unhealthy", "🟣 Purple"),
        6: ("Hazardous", "☠️ Brown")
    }
    return mapping.get(int(aqi_class), ("Unknown", "⚠️ Grey"))


# ------------------------- MAIN APP ------------------------- #
def app():
    st.title("🌫️ AQI Forecast (Next 3 Days)")

    os.makedirs("data", exist_ok=True)
    local_cache = "data/recent_snapshot.csv"

    # ------------------- Load latest features ------------------- #
    try:
        project, fs = connect_hopsworks()
        fg = fs.get_feature_group("computed_features_historical_v3", version=1)

        df = fg.read()
        if df is None or df.empty:
            raise ValueError("Feature group returned empty or None.")

        df = df.sort_values("datetime").tail(10)
        df.to_csv(local_cache, index=False)
        st.success("✅ Loaded recent observation snapshot from Hopsworks.")

    except Exception as e:
        st.warning(f"⚠️ Could not fetch data from Hopsworks: {e}")

        if os.path.exists(local_cache):
            df = pd.read_csv(local_cache)
            st.info("📁 Loaded cached snapshot from previous run.")
        else:
            st.error("❌ No cached data found. Please connect once to Hopsworks to cache it.")
            return

    # ✅ Display recent data
    st.dataframe(df, width='stretch')

    # ------------------- Load model ------------------- #
    model = load_model()

    # ------------------- Prepare model input ------------------- #
    # Keep only numeric columns
    feature_input = df.select_dtypes(include=["number"])
    feature_input = feature_input.drop(columns=["target", "label"], errors="ignore")

    # ------------------- Align with model features ------------------- #
    try:
        model_features = list(model.feature_names_in_)
        st.write("🧩 Model expects:", model_features)
        st.write("📊 Input columns:", feature_input.columns.tolist())

        # Add any missing columns with default value (0)
        for col in model_features:
            if col not in feature_input.columns:
                feature_input[col] = 0

        # Keep only the columns model expects (in correct order)
        feature_input = feature_input[model_features]

    except Exception as e:
        st.warning(f"⚠️ Could not align features automatically: {e}")

    # ------------------- Predict AQI ------------------- #
    if st.button("🚀 Predict AQI (Next 3 Days)", width='stretch'):
        st.info("⏳ Generating forecast...")
        try:
            preds = model.predict(feature_input)
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            return

        # ✅ Create 3-day forecast
        future_dates = [(datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(3)]
        preds = preds[:3] if len(preds) >= 3 else list(preds) + [preds[-1]] * (3 - len(preds))

        results = pd.DataFrame({
            "Date": future_dates,
            "Predicted AQI Class": preds,
            "Condition": [aqi_alert(p)[0] for p in preds],
            "Alert": [aqi_alert(p)[1] for p in preds]
        })

        # ✅ Show results
        st.success("✅ Forecast Ready!")
        st.dataframe(results, width='stretch')

        # ✅ Plot forecast trend
        fig = px.line(
            results,
            x="Date",
            y="Predicted AQI Class",
            markers=True,
            title="🌥️ Predicted AQI Trend (Next 3 Days)"
        )
        st.plotly_chart(fig, width='stretch')

        # ✅ Display alerts
        st.subheader("⚠️ Air Quality Alerts")
        for _, row in results.iterrows():
            st.info(f"**{row['Date']} → {row['Condition']}** ({row['Alert']})")


# ------------------------- RUN APP ------------------------- #
if __name__ == "__main__":
    app()
