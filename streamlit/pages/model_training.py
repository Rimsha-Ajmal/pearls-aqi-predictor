import streamlit as st
import pandas as pd
import subprocess
import os
import json
import time

# ---------- Paths ----------
ALL_METRICS_PATH = "models/all_models_metrics.json"
LAST_LOG_PATH = "models/last_training_log.txt"
os.makedirs("models", exist_ok=True)

# ---------- Helper Functions ----------
def run_training_script():
    """Runs the training script and streams logs live in Streamlit."""
    cmd = ["python", "scripts/train_model.py"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log_box = st.empty()
    logs = ""

    while True:
        output = process.stdout.readline()
        if output:
            logs += output
            log_box.code(logs, language="bash")
        if process.poll() is not None:
            break
        time.sleep(0.05)

    with open(LAST_LOG_PATH, "w", encoding="utf-8") as f:
        f.write(logs)

    return process.returncode, logs


def load_all_metrics():
    """Load all model metrics from local file."""
    if os.path.exists(ALL_METRICS_PATH):
        with open(ALL_METRICS_PATH, "r") as f:
            return json.load(f)
    return []


def show_local_fallback():
    """Show fallback message when working offline or training fails."""
    if os.path.exists(LAST_LOG_PATH):
        with open(LAST_LOG_PATH, "r", encoding="utf-8") as f:
            logs = f.read()
        st.warning("⚠️ Showing locally saved training results (offline mode).")
        st.code(logs, language="bash")


# ---------- Streamlit App ----------
def app():
    st.title("🤖 Train AQI Prediction Models")
    st.write("Train multiple models, compare their performance, and automatically select the best one.")

    # --- Training button ---
    if st.button("🟩 Start Training"):
        with st.spinner("⏳ Training models..."):
            returncode, logs = run_training_script()

        if returncode == 0:
            st.success("✅ Training completed successfully!")
            st.code(logs, language="bash")
        else:
            st.error("❌ Training failed. Showing last saved results instead.")
            show_local_fallback()

        st.rerun()

    st.divider()
    st.header("📊 Latest Training Results")

    # --- Load all models metrics ---
    all_metrics = load_all_metrics()
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        # Sort by best R²
        df = df.sort_values("R2", ascending=False).reset_index(drop=True)

        # Display comparison table
        st.markdown("### 🚀 Model Comparison:")
        st.dataframe(df, use_container_width=True)

        # Highlight best model
        best_model = df.iloc[0]
        st.markdown(
            f"🏆 **Best Model:** `{best_model['Model']}`\n\n"
            f"📈 **Metrics** → MAE: `{best_model['MAE']:.2f}`, "
            f"RMSE: `{best_model['RMSE']:.2f}`, "
            f"R²: `{best_model['R2']:.3f}`"
        )

    else:
        st.warning("⚠️ No metrics found. Run training first or load offline results.")
        show_local_fallback()
