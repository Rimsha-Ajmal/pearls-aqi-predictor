#!/usr/bin/env python3
# ==========================================================
#  AQI Model Incremental Retraining Script
#  Author: Rimsha Ajmal
#  Description:
#    Automatically retrains your RandomForest AQI model
#    using new hourly features from Hopsworks.
# ==========================================================

import os
import sys
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import hopsworks
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from hsml.schema import Schema
from hsml.model_schema import ModelSchema


# ------------------------------
# 1️⃣ Connect to Hopsworks
# ------------------------------
print("🔗 Connecting to Hopsworks...")
project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()
mr = project.get_model_registry()
print("✅ Connected to project:", project.name)


# ------------------------------
# 2️⃣ Load feature group
# ------------------------------
fg = fs.get_feature_group("model_features", version=None)
df = fg.read()
print("✅ Loaded feature data. Shape:", df.shape)

# Ensure datetime exists
if "datetime" not in df.columns:
    raise ValueError("❌ 'datetime' column not found in feature group")

df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)


# ------------------------------
# 3️⃣ Verify feature ingestion continuity
# ------------------------------
time_diffs = df["datetime"].diff().dropna()
avg_interval = time_diffs.mean().total_seconds() / 3600
print(f"⏱️ Average ingestion interval: {avg_interval:.2f} hours")

if avg_interval > 2:
    print("⚠️ Warning: Feature ingestion not hourly (avg > 2h). Check pipeline.")
else:
    print("✅ Feature ingestion seems consistent (hourly or better).")


# ------------------------------
# 4️⃣ Load last checkpoint
# ------------------------------
checkpoint_file = "last_datetime.txt"
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, "r") as f:
        last_dt = pd.to_datetime(f.read().strip())
    print(f"ℹ️ Last checkpoint found at: {last_dt}")
else:
    print("ℹ️ No checkpoint found — starting fresh.")
    last_dt = pd.Timestamp.min


# ------------------------------
# 5️⃣ Retraining frequency limiter
# ------------------------------
now_utc = datetime.now(timezone.utc)
if (now_utc - last_dt).total_seconds() < 6 * 3600 and last_dt != pd.Timestamp.min:
    print("⏳ Skipping retraining: last update was within 6 hours.")
    sys.exit(0)  # ✅ clean exit (no warning)


# ------------------------------
# 6️⃣ Load only new data
# ------------------------------
df_new = df[df["datetime"] > last_dt].copy()
if df_new.empty:
    print("🚫 No new data to retrain. Exiting gracefully.")
    print(f"ℹ️ Last checkpoint remains at: {last_dt}")
    sys.exit(0)  # ✅ clean exit (no warning)

print(f"✅ New data to train: {df_new.shape[0]} rows")


# ------------------------------
# 7️⃣ Convert scaled AQI → numeric AQI
# ------------------------------
aqi_scale_map = {1: 50, 2: 100, 3: 150, 4: 200, 5: 300}
aqi_cols = [c for c in df_new.columns if "aqi" in c.lower()]
for col in aqi_cols:
    df_new[col] = df_new[col].map(aqi_scale_map).fillna(df_new[col])
print(f"🔄 Converted scaled AQI columns to numeric AQI for {len(aqi_cols)} columns")


# ------------------------------
# 8️⃣ Prepare dataset
# ------------------------------
H = 72
target_col = f"aqi_t_plus_{H}"

if target_col not in df_new.columns:
    raise ValueError(f"❌ Target column {target_col} not found in dataset.")

df_sup = df_new.dropna(subset=[target_col]).copy()
non_feature_cols = ["datetime", "timestamp"]
features = [c for c in df_sup.columns if c not in non_feature_cols + [target_col]]

X = df_sup[features].replace([np.inf, -np.inf], np.nan)
X[X <= 0] = np.nan
X = X.ffill().dropna()

y = df_sup.loc[X.index, target_col].astype(float)

split_frac = 0.8
split_idx = int(len(X) * split_frac)
X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
print(f"✅ Data prepared | Train: {len(X_train)}, Val: {len(X_val)}")


# ------------------------------
# 9️⃣ Train RandomForest
# ------------------------------
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

def get_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

y_pred = rf.predict(X_val)
mae, rmse, r2 = get_metrics(y_val, y_pred)
print(f"📈 Validation -> MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")


# ------------------------------
# 🔟 Compare with existing model
# ------------------------------
model_name = f"AQI_RandomForest_H{H}"
better = False

try:
    existing_model = mr.get_model(model_name, version=None)
    old = existing_model.to_dict().get("metrics", {})
    old_mae, old_rmse, old_r2 = float(old.get("mae", 999)), float(old.get("rmse", 999)), float(old.get("r2", -999))
    print(f"📦 Existing model v{existing_model.version}: MAE={old_mae:.4f}, RMSE={old_rmse:.4f}, R²={old_r2:.4f}")

    improved = sum([
        mae < old_mae * 0.98,
        rmse < old_rmse * 0.98,
        r2 > old_r2 * 1.01
    ])

    if improved >= 2 and r2 >= old_r2:
        better = True
        print("✅ New model shows improvement — registering new version.")
    elif abs(mae - old_mae) < 0.001 and abs(rmse - old_rmse) < 0.001 and abs(r2 - old_r2) < 0.001:
        print("ℹ️ Model performance unchanged — skipping registration.")
    else:
        print("⚠️ No significant improvement — keeping existing version.")
except Exception:
    print("ℹ️ No existing model found — registering first version.")
    better = True


# ------------------------------
# 1️⃣1️⃣ Register model if better
# ------------------------------
if better:
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{model_name}.pkl"
    joblib.dump(rf, model_path)

    input_schema = Schema(X_train)
    output_schema = Schema(y_train)
    model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)

    model = mr.sklearn.create_model(
        name=model_name,
        metrics={"mae": mae, "rmse": rmse, "r2": r2},
        model_schema=model_schema,
        description=f"RandomForest AQI predictor ({H}-hour horizon)"
    )
    model.save(model_path)
    print(f"✅ Registered new model '{model_name}' with MAE={mae:.4f}, R²={r2:.4f}")
else:
    print("🚫 Model not registered — using existing version.")


# ------------------------------
# 1️⃣2️⃣ Update checkpoint
# ------------------------------
new_last_dt = df["datetime"].max()
with open(checkpoint_file, "w") as f:
    f.write(str(new_last_dt))

if new_last_dt > last_dt:
    print(f"🔖 Updated checkpoint: {new_last_dt} ✅ (moved forward)")
else:
    print(f"⚠️ Checkpoint not advanced: last={last_dt}, new={new_last_dt}")
