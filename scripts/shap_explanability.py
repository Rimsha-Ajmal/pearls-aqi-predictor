# """
# SHAP Explainability for RandomForest AQI Model (Hopsworks)
# """

# # ------------------------- IMPORT LIBRARIES -------------------------
# import shap
# import joblib
# import pandas as pd
# import matplotlib.pyplot as plt
# import hopsworks
# # Install dotenv if not installed
# # pip install python-dotenv

# from dotenv import load_dotenv
# import os

# load_dotenv()
# HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")


# # ------------------------- CONFIGURATION -------------------------
# MODEL_NAME = "randomForest_test_3_model"
# MODEL_VERSION = 1   
# DATA_PATH = "computed_features_historical_selected_cleaned.csv"
# OUTPUT_CSV = "shap_feature_importance.csv"
# EXCLUDE_COLS = ['datetime']

# # ------------------------- CONNECT TO HOPSWORKS -------------------------

# project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
# fs = project.get_feature_store()
# print("✅ Connected to Hopsworks project successfully!")

# # ------------------------- LOAD DATA -------------------------
# df = pd.read_csv(DATA_PATH)
# print("✅ Data loaded successfully!")

# # ------------------------- LOAD MODEL -------------------------
# print("🔗 Connecting to Hopsworks model registry...")
# mr = project.get_model_registry()
# model_obj = mr.get_model(MODEL_NAME, version=MODEL_VERSION)
# model_dir = model_obj.download()
# model_path = model_dir + "/randomForest_model.pkl"
# rf_model = joblib.load(model_path)
# print("✅ RandomForest model loaded!")

# # ------------------------- PREPARE TEST DATA -------------------------
# X_test = df.drop(columns=[c for c in EXCLUDE_COLS if c in df.columns], errors='ignore')
# print(f"✅ Test dataset loaded: {X_test.shape}")

# # ------------------------- COMPUTE SHAP VALUES -------------------------
# print("⚙️ Computing SHAP values...")
# explainer = shap.TreeExplainer(rf_model)
# shap_values = explainer.shap_values(X_test)

# # ------------------------- GLOBAL FEATURE IMPORTANCE -------------------------
# print("📊 Generating SHAP global importance (bar)...")
# shap.summary_plot(shap_values, X_test, plot_type="bar")
# plt.tight_layout()
# plt.savefig("shap_global_importance_bar.png", dpi=300)
# plt.show()

# # ------------------------- GLOBAL FEATURE IMPACT -------------------------
# print("🌈 Generating SHAP beeswarm plot...")
# shap.summary_plot(shap_values, X_test)
# plt.tight_layout()
# plt.savefig("shap_beeswarm_plot.png", dpi=300)
# plt.show()

# # ------------------------- SAVE SHAP IMPORTANCE TO CSV -------------------------
# print("💾 Saving ranked feature importances...")
# shap_importance = pd.DataFrame({
#     'feature': X_test.columns,
#     'mean_abs_shap_value': abs(shap_values).mean(axis=0)
# }).sort_values(by='mean_abs_shap_value', ascending=False)

# shap_importance.to_csv(OUTPUT_CSV, index=False)
# print(f"✅ SHAP feature importance saved to: {OUTPUT_CSV}")

# # ------------------------- LOCAL EXPLANATION FOR ONE SAMPLE -------------------------
# print("🔍 Local explanation for one sample...")
# sample_idx = 0
# shap.force_plot(
#     explainer.expected_value,
#     shap_values[sample_idx],
#     X_test.iloc[sample_idx],
#     matplotlib=True
# )
# plt.show()

# # ------------------------- TOP-N FEATURES BY SHAP -------------------------
# top_n = 20
# top_features = shap_importance.head(top_n)['feature'].tolist()
# print(f"🏆 Top {top_n} features by SHAP importance:")
# print(top_features)

# SHAP_JSON_PATH = "models/shap_feature_importance.json"

# # Convert to dict for JSON
# shap_dict = dict(zip(shap_importance['feature'], shap_importance['mean_abs_shap_value']))

# # Save as JSON
# with open(SHAP_JSON_PATH, "w") as f:
#     json.dump(shap_dict, f, indent=2)

# print(f"✅ SHAP feature importance saved to JSON: {SHAP_JSON_PATH}")
