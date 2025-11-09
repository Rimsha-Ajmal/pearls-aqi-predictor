# train_and_evaluate_model.py

import os
import joblib
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from hsml.schema import Schema
from hsml.model_schema import ModelSchema
from dotenv import load_dotenv

load_dotenv()  # Load .env file

# ==============================
# 1. CONNECT TO HOPSWORKS
# ==============================
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
fs, mr = None, None

if HOPSWORKS_API_KEY:
    try:
        import hopsworks
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
        fs = project.get_feature_store()
        mr = project.get_model_registry()
        print(f"✅ Connected to project: {project.name}")
    except Exception as e:
        print(f"⚠️ Could not connect to Hopsworks. Running offline. Error: {e}")
else:
    print("⚠️ HOPSWORKS_API_KEY not set. Running offline mode.")

# ==============================
# 2. LOAD FEATURE GROUP
# ==============================
if fs is not None:
    fg = fs.get_feature_group("computed_features_historical", version=1)
    df = fg.read()
    print("✅ Loaded feature data from Hopsworks. Shape:", df.shape)
else:
    # Offline fallback: load from local CSV if available
    offline_path = "data/computed_features_historical.csv"
    if os.path.exists(offline_path):
        df = pd.read_csv(offline_path)
        print("⚠️ Loaded feature data from local CSV. Shape:", df.shape)
    else:
        raise FileNotFoundError("❌ No Hopsworks connection and local CSV not found.")

# ==============================
# 3. PREPARE DATA
# ==============================
H = 72  # forecast horizon in hours
target_col = f"aqi_t_plus_{H}"

if target_col not in df.columns:
    raise ValueError(f"❌ Target column '{target_col}' not found in feature group.")

df_sup = df.dropna(subset=[target_col]).copy()
non_feature_cols = ["datetime", "timestamp"]
features = [c for c in df_sup.columns if c not in non_feature_cols + [target_col]]
print(f"📊 Total features before cleaning: {len(features)}")

X = df_sup[features].replace([np.inf, -np.inf], np.nan)
X[X <= 0] = np.nan
X = X.ffill().dropna()
y = df_sup.loc[X.index, target_col].astype(float)
print(f"🧹 Cleaned missing values. Remaining samples: {len(X)}")

# ==============================
# 4. TRAIN / TEST SPLIT
# ==============================
split_frac = 0.8
split_idx = int(len(X) * split_frac)
X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]
print(f"✅ Training on first 80%: {len(X_train)} samples, Test: {len(X_test)} samples")

# ==============================
# 5. TRAIN & EVALUATE MODELS
# ==============================
def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

models = {
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "Ridge": Ridge(alpha=1.0, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42)
}

results = []

for name, model in models.items():
    print(f"\n🚀 Training {name} ...")
    model.fit(X_train, y_train)
    preds = model.predict(X_train)
    mae, rmse, r2 = metrics(y_train, preds)
    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
    print(f"{name} -> MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")

results_df = pd.DataFrame(results).sort_values("RMSE")
best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]
best_metrics = results_df.iloc[0]

print(f"\n🏆 Best Model: {best_model_name}")
print(f"📈 Metrics -> MAE: {best_metrics['MAE']:.3f}, RMSE: {best_metrics['RMSE']:.3f}, R²: {best_metrics['R2']:.3f}")

# ==============================
# 6. SAVE & REGISTER BEST MODEL
# ==============================
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "randomForest_model.pkl")
joblib.dump(best_model, model_path)
print(f"💾 Model saved locally at: {model_path}")

if mr is not None:
    input_schema = Schema(X_train)
    output_schema = Schema(y_train)
    model_schema = ModelSchema(input_schema, output_schema)
    model_hf = mr.sklearn.create_model(
        name="randomForest_shap_30_model",
        metrics={
            "mae": float(best_metrics["MAE"]),
            "rmse": float(best_metrics["RMSE"]),
            "r2": float(best_metrics["R2"]),
        },
        model_schema=model_schema,
        description=f"{best_model_name} trained on top 30 SHAP features for 3-day AQI prediction (Karachi, hourly)"
    )
    model_hf.save(model_path)
    print(f"✅ Registered model 'randomForest_shap_30_model' in Hopsworks (version {model_hf.version})")

# ==============================
# 7. SAVE METRICS & LOGS FOR STREAMLIT
# ==============================
metrics_path = os.path.join("models", "all_models_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(results, f, indent=4)
print(f"📊 Saved all models metrics to {metrics_path}")

log_path = os.path.join("models", "last_training_log.txt")
with open(log_path, "w") as f:
    f.write("Training completed successfully.\n")
    f.write(str(results))
print(f"💾 Saved last training log to {log_path}")
