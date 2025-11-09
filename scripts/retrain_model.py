# ========================================================== 
# AQI Model Incremental Retraining Script (Hopsworks Checkpoint Version)
# ==========================================================

import os
import joblib
import pandas as pd
import numpy as np
from datetime import timezone
import hopsworks
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from hsml.schema import Schema
from hsml.model_schema import ModelSchema

# ------------------------------ 1️⃣ Connect to Hopsworks ------------------------------
print("🔗 Connecting to Hopsworks...")
project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()
mr = project.get_model_registry()
print("✅ Connected to project:", project.name)

# ------------------------------ 2️⃣ Load feature groups ------------------------------
fg_hist = fs.get_feature_group("computed_features_historical", version=1)
fg_realtime = fs.get_feature_group("computed_features_realtime", version=1)

df_hist = fg_hist.read()
df_realtime = fg_realtime.read()

# Ensure datetime columns are UTC-aware
df_hist["datetime"] = pd.to_datetime(df_hist["datetime"], utc=True)
df_realtime["datetime"] = pd.to_datetime(df_realtime["datetime"], utc=True)

# Combine and remove duplicates
df_all = pd.concat([df_hist, df_realtime]).drop_duplicates(subset=["datetime"])
df_all = df_all.sort_values("datetime").reset_index(drop=True)
print(f"✅ Combined feature data shape: {df_all.shape}")

# ------------------------------ 3️⃣ Load or initialize checkpoint from Hopsworks ------------------------------
checkpoint_fg_name = "retraining_checkpoint"

try:
    checkpoint_fg = fs.get_feature_group(name=checkpoint_fg_name, version=1)
    df_checkpoint = checkpoint_fg.read()
    last_dt = pd.to_datetime(
        df_checkpoint.loc[df_checkpoint.pipeline_name=="aqi_model", "last_datetime"].values[0], utc=True
    )
    print(f"ℹ️ Last checkpoint from Hopsworks: {last_dt}")
except Exception:
    # Feature group doesn't exist, create it
    checkpoint_fg = fs.create_feature_group(
        name=checkpoint_fg_name,
        version=1,
        primary_key=["pipeline_name"],
        description="Tracks last processed datetime for AQI retraining",
        online_enabled=False
    )
    last_dt = pd.Timestamp.min.replace(tzinfo=timezone.utc)
    print("ℹ️ No checkpoint found — starting fresh (min timestamp UTC).")

# ------------------------------ 4️⃣ Filter for new data ------------------------------
df_new = df_all[df_all["datetime"] > last_dt].copy()
if df_new.empty:
    print("🚫 No new data to retrain. ✅ Pipeline ran successfully.")
else:
    print(f"✅ Found {df_new.shape[0]} new rows for retraining.")

    # ------------------------------ 5️⃣ Prepare supervised dataset ------------------------------
    H = 72
    target_col = f"aqi_t_plus_{H}"
    non_feature_cols = ["datetime", "timestamp"]

    df_sup = df_new.dropna(subset=[target_col]).copy()
    features = [c for c in df_sup.columns if c not in non_feature_cols + [target_col]]

    X = df_sup[features].replace([np.inf, -np.inf], np.nan)
    X[X <= 0] = np.nan
    X = X.ffill().dropna()
    y = df_sup.loc[X.index, target_col].astype(float)

    if len(X) < 100:
        print("⚠️ Not enough new data for retraining. ✅ Pipeline ran successfully.")
    else:
        split_frac = 0.8
        split_idx = int(len(X) * split_frac)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"✅ Data ready | Train: {len(X_train)}, Val: {len(X_val)}")

        # ------------------------------ 6️⃣ Load existing model ------------------------------
        model_name = "randomForest_shap_30_model"
        existing_model = None
        try:
            existing_model = mr.get_model(model_name, version=None)
            old_metrics = existing_model.to_dict().get("metrics", {})
            old_mae = float(old_metrics.get("mae", 999))
            old_rmse = float(old_metrics.get("rmse", 999))
            old_r2 = float(old_metrics.get("r2", -999))
            print(f"📦 Existing model v{existing_model.version}: MAE={old_mae}, RMSE={old_rmse}, R²={old_r2}")
        except Exception:
            print("ℹ️ No existing model found — will register first version.")

        # ------------------------------ 7️⃣ Train new RandomForest ------------------------------
        rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        print("✅ Model retrained successfully.")

        # ------------------------------ 8️⃣ Evaluate ------------------------------
        def get_metrics(y_true, y_pred):
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            return mae, rmse, r2

        y_pred = rf.predict(X_val)
        mae, rmse, r2 = get_metrics(y_val, y_pred)
        print(f"📈 Validation -> MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

        # ------------------------------ 9️⃣ Compare & Register ------------------------------
        better = False
        if existing_model:
            improved_count = sum([
                mae < old_mae * 0.98,
                rmse < old_rmse * 0.98,
                r2 > old_r2 * 1.01
            ])
            if improved_count >= 2 and r2 >= old_r2:
                better = True
                print("✅ New model improved — registering new version.")
            else:
                print("⚠️ No improvement — retaining existing model.")
        else:
            better = True

        if better:
            os.makedirs("models", exist_ok=True)
            model_path = f"models/{model_name}.pkl"
            joblib.dump(rf, model_path)
            print(f"💾 Saved model locally at {model_path}")

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
            print("✅ Registered new model version in Hopsworks.")

        # ------------------------------ 🔟 Update checkpoint in Hopsworks & save locally ------------------------------
        new_last_dt = df_all["datetime"].max()
        checkpoint_df = pd.DataFrame([{"pipeline_name": "aqi_model", "last_datetime": new_last_dt}])
        checkpoint_fg.insert(checkpoint_df, write_options={"upsert": True})
        print(f"🔖 Checkpoint updated in Hopsworks → {new_last_dt}")

        # Save local checkpoint for development/testing
        with open("last_datetime_local.txt", "w") as f:
            f.write(str(new_last_dt))
        print(f"💾 Local checkpoint saved → last_datetime_local.txt")

        print("🎯 Pipeline completed successfully.")
