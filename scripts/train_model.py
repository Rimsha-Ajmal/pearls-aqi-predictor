# train_and_evaluate_model.py

import os
import joblib
import pandas as pd
import numpy as np
import hopsworks
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from hsml.schema import Schema
from hsml.model_schema import ModelSchema

# ==============================
# 1. CONNECT TO HOPSWORKS
# ==============================
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
if not HOPSWORKS_API_KEY:
    raise ValueError("❌ HOPSWORKS_API_KEY environment variable not set!")

print("🔐 API key loaded successfully.")
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()
mr = project.get_model_registry()
print(f"✅ Connected to project: {project.name}")

# ==============================
# 2. LOAD FEATURE GROUP
# ==============================
fg = fs.get_feature_group("computed_features_historical_v3", version=1)
df = fg.read()
print("✅ Loaded feature data. Shape:", df.shape)

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

# Clean missing values
X = df_sup[features].copy()
X = X.replace([np.inf, -np.inf], np.nan)
X[X <= 0] = np.nan
X = X.ffill()
missing_before = X.isna().sum().sum()
X = X.dropna()
missing_after = X.isna().sum().sum()
print(f"🧹 Cleaned missing values: {missing_before} → {missing_after}")

# Align target
y = df_sup.loc[X.index, target_col].astype(float)

# ==============================
# 4. TRAIN / TEST SPLIT
# ==============================
split_frac = 0.8
split_idx = int(len(X) * split_frac)
X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]

print(f"✅ Training on first 80% of data: {len(X_train)} samples")
print(f"⏸ 20% reserved for later testing: {len(X_test)} samples")

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
    print(f"{name} -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")

results_df = pd.DataFrame(results).sort_values(by="RMSE")
print("\n📊 Model Comparison:\n", results_df)

best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]
best_metrics = results_df.iloc[0]
print(f"\n🏆 Best Model: {best_model_name}")
print(f"📈 Metrics -> MAE: {best_metrics['MAE']:.2f}, RMSE: {best_metrics['RMSE']:.2f}, R²: {best_metrics['R2']:.3f}")

# ==============================
# 6. SAVE & REGISTER BEST MODEL
# ==============================
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "randomForest_model.pkl")
joblib.dump(best_model, model_path)
print(f"💾 Model saved locally at: {model_path}")

input_schema = Schema(X_train)
output_schema = Schema(y_train)
model_schema = ModelSchema(input_schema, output_schema)

model = mr.sklearn.create_model(
    name="randomForest_test_3_model",
    metrics={
        "mae": float(best_metrics["MAE"]),
        "rmse": float(best_metrics["RMSE"]),
        "r2": float(best_metrics["R2"]),
    },
    model_schema=model_schema,
    description=f"{best_model_name} model trained for AQI prediction ({H}-hour horizon)"
)
model.save(model_path)
print(f"\n✅ Registered model 'randomForest_model' in Hopsworks (version {model.version})")

# ==============================
# 7. TESTING (20%)
# ==============================
USE_ZERO_FOR_NON_SELECTED = True

def safe_metrics(y_true, y_pred):
    if len(y_true) == 0:
        return (np.nan, np.nan, np.nan)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred) if len(y_true) >= 2 else np.nan
    return (mae, rmse, r2)

# Load model from registry
model = mr.get_model("randomForest_test_3_model", version=1)
model_dir = model.download()
rf_model = joblib.load(model_dir + "/randomForest_model.pkl")
print("✅ Model loaded for testing!")

# Read feature group again
df = fg.read()
df = df.dropna(subset=[target_col])
split_idx = int(len(df) * 0.8)
df_test = df.iloc[split_idx:].reset_index(drop=True)
non_feature_cols = ["datetime", "timestamp"]
features = [c for c in df_test.columns if c not in non_feature_cols + [target_col]]
X_full = df_test[features].copy()
y_full = df_test[target_col].copy()

print("✅ Test set:", X_full.shape)

# ==============================
# 8. FEATURE IMPORTANCE
# ==============================
feature_importances = pd.Series(
    rf_model.feature_importances_,
    index=features
).sort_values(ascending=False)
print("\n🔥 Top 10 Features:\n", feature_importances.head(10))

# ==============================
# 9. EVALUATE FEATURE COUNTS
# ==============================
step = 5
max_feats = len(features)
feature_counts = list(range(step, max_feats + 1, step))
if feature_counts[-1] != max_feats:
    feature_counts.append(max_feats)

print("\nWill evaluate feature counts:", feature_counts)

# AQI-level bins (data-driven)
q_low = np.nanpercentile(y_full, 33.3333)
q_high = np.nanpercentile(y_full, 66.6667)
print(f"\nAQI tertile cutpoints: low<= {q_low:.4f}, high> {q_high:.4f}")

def aqi_level_mask(y, level):
    if level == "low":
        return y <= q_low
    elif level == "normal":
        return (y > q_low) & (y <= q_high)
    elif level == "high":
        return y > q_high
    else:
        raise ValueError("level must be 'low','normal','high'")

results = []
for k in feature_counts:
    top_k = feature_importances.head(k).index.tolist()
    X_test = X_full.copy()

    if USE_ZERO_FOR_NON_SELECTED:
        for col in X_test.columns:
            if col not in top_k:
                X_test[col] = 0.0
    else:
        col_means = X_test.mean()
        for col in X_test.columns:
            if col not in top_k:
                X_test[col] = col_means[col]

    y_pred = rf_model.predict(X_test)
    mae_all, rmse_all, r2_all = safe_metrics(y_full.values, y_pred)

    metrics_by_level = {}
    for lvl in ("low", "normal", "high"):
        mask = aqi_level_mask(y_full, lvl)
        y_true_lvl = y_full[mask].values
        y_pred_lvl = y_pred[mask]
        mae_lvl, rmse_lvl, r2_lvl = safe_metrics(y_true_lvl, y_pred_lvl)
        metrics_by_level[f"MAE_{lvl}"] = mae_lvl
        metrics_by_level[f"RMSE_{lvl}"] = rmse_lvl
        metrics_by_level[f"R2_{lvl}"] = r2_lvl

    results.append({
        "Features": k,
        "MAE": mae_all,
        "RMSE": rmse_all,
        "R2": r2_all,
        **metrics_by_level
    })

    print(f"✅ Top {k}: MAE={mae_all:.3f}, RMSE={rmse_all:.3f}, R²={r2_all:.4f} "
          f"(low RMSE={metrics_by_level['RMSE_low']:.3f}, normal RMSE={metrics_by_level['RMSE_normal']:.3f}, high RMSE={metrics_by_level['RMSE_high']:.3f})")

# ==============================
# 10. SAVE RESULTS & PLOTS
# ==============================
results_df = pd.DataFrame(results).sort_values("Features").reset_index(drop=True)
results_df.to_csv("feature_count_evaluation_results.csv", index=False)
print("\n📊 Saved results to feature_count_evaluation_results.csv")

plt.figure(figsize=(9,5))
plt.plot(results_df["Features"], results_df["RMSE"], marker='o')
plt.title("Overall RMSE vs Number of Top Features")
plt.xlabel("Number of Top Features")
plt.ylabel("RMSE")
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(9,5))
plt.plot(results_df["Features"], results_df["RMSE_low"], marker='o', label="Low AQI RMSE")
plt.plot(results_df["Features"], results_df["RMSE_normal"], marker='o', label="Normal AQI RMSE")
plt.plot(results_df["Features"], results_df["RMSE_high"], marker='o', label="High AQI RMSE")
plt.title("Class-wise RMSE vs Number of Top Features")
plt.xlabel("Number of Top Features")
plt.ylabel("RMSE")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# ==============================
# 11. BEST FEATURE COUNTS
# ==============================
best_row_overall = results_df.loc[results_df["RMSE"].idxmin()]
print(f"\n🏆 Best Feature Count (overall RMSE): {int(best_row_overall['Features'])}")
print(best_row_overall)

for level in ["low", "normal", "high"]:
    best = results_df.loc[results_df[f"RMSE_{level}"].idxmin()] if results_df[f"RMSE_{level}"].notnull().any() else None
    if best is not None:
        print(f"\n🏆 Best Feature Count for {level.upper()} AQI (RMSE): {int(best['Features'])}")
        print(best)

best_k = int(best_row_overall["Features"])
print(f"\nTop {best_k} features:\n", feature_importances.head(best_k).index.tolist())
