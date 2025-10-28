"""
train_aqi_model.py
------------------
This script connects to Hopsworks, loads AQI feature data,
trains multiple regression models (RandomForest, Ridge, GradientBoosting),
evaluates them, and registers the best model in the Hopsworks Model Registry.
"""

import os
import joblib
import pandas as pd
import numpy as np
import hopsworks
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from hsml.schema import Schema
from hsml.model_schema import ModelSchema

# ----------------------------------------
# 1. CONNECT TO HOPSWORKS
# ----------------------------------------
print("🔗 Connecting to Hopsworks...")
project = hopsworks.login()
fs = project.get_feature_store()
mr = project.get_model_registry()
print(f"✅ Connected to project: {project.name}")

# ----------------------------------------
# 2. LOAD FEATURE GROUP
# ----------------------------------------
print("📥 Loading feature group 'model_features'...")
fg = fs.get_feature_group("model_features", version=1)
df = fg.read()
print(f"✅ Loaded feature data. Shape: {df.shape}")

# ----------------------------------------
# 3. AQI VALUE CONVERSION
# ----------------------------------------
print("🔄 Converting scaled AQI columns to actual numeric AQI values...")

aqi_scale_map = {
    1: 50,   # Good
    2: 100,  # Moderate
    3: 150,  # Unhealthy for Sensitive Groups
    4: 200,  # Unhealthy
    5: 300,  # Very Unhealthy
}

aqi_cols = [c for c in df.columns if 'aqi' in c.lower()]
for col in aqi_cols:
    df[col] = df[col].map(aqi_scale_map).fillna(df[col])

print(f"✅ Converted {len(aqi_cols)} AQI-related columns.")

# ----------------------------------------
# 4. DATA PREPARATION
# ----------------------------------------
H = 72  # Forecast horizon in hours
target_col = f"aqi_t_plus_{H}"

if target_col not in df.columns:
    raise ValueError(f"❌ Target column '{target_col}' not found in feature group.")

df_sup = df.dropna(subset=[target_col]).copy()

non_feature_cols = ["datetime", "timestamp"]
features = [c for c in df_sup.columns if c not in non_feature_cols + [target_col]]
print(f"📊 Total features before cleaning: {len(features)}")

# Clean invalid values
X = df_sup[features].copy()
X = X.replace([np.inf, -np.inf], np.nan)
X[X <= 0] = np.nan
X = X.ffill()
missing_before = X.isna().sum().sum()
X = X.dropna()
missing_after = X.isna().sum().sum()
print(f"🧹 Cleaned missing values: {missing_before} → {missing_after}")

y = df_sup.loc[X.index, target_col].astype(float)

# Train/validation split
split_frac = 0.8
split_idx = int(len(X) * split_frac)
X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
print(f"✅ Data prepared | Train: {len(X_train)}, Val: {len(X_val)}")

# ----------------------------------------
# 5. TRAIN & EVALUATE MODELS
# ----------------------------------------
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
    preds = model.predict(X_val)
    mae, rmse, r2 = metrics(y_val, preds)
    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
    print(f"{name} -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")

results_df = pd.DataFrame(results).sort_values(by="RMSE")
print("\n📊 Model Comparison:\n", results_df)

# ----------------------------------------
# 6. REGISTER BEST MODEL IN HOPSWORKS
# ----------------------------------------
best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]
best_metrics = results_df.iloc[0]

print(f"\n🏆 Best Model: {best_model_name}")
print(f"📈 Metrics -> MAE: {best_metrics['MAE']:.2f}, RMSE: {best_metrics['RMSE']:.2f}, R²: {best_metrics['R2']:.3f}")

# Save model locally
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, f"{best_model_name}_H{H}.pkl")
joblib.dump(best_model, model_path)
print(f"💾 Model saved locally at: {model_path}")

# Create model schema
input_schema = Schema(X_train)
output_schema = Schema(y_train)
model_schema = ModelSchema(input_schema, output_schema)

# Register the best model
model = mr.sklearn.create_model(
    name=f"AQI_{best_model_name}_H{H}",
    metrics={
        "mae": float(best_metrics["MAE"]),
        "rmse": float(best_metrics["RMSE"]),
        "r2": float(best_metrics["R2"]),
    },
    model_schema=model_schema,
    description=f"{best_model_name} model trained for exact AQI prediction ({H}-hour horizon)"
)
model.save(model_path)

print(f"\n✅ Registered model '{best_model_name}' in Hopsworks (version {model.version})")
print("🎯 Training and registration complete!")
