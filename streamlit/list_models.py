from utils.hops import connect_hopsworks

print("\n🔗 Connecting to Hopsworks...\n")

project, _ = connect_hopsworks()
mr = project.get_model_registry()

print("\n📦 Available models in your Hopsworks registry:\n")    

try:
    model = mr.get_model("AQI_RandomForest_H72", version=14)
    print("✅ Model found:", model.name, "version", model.version)
    print("📁 Downloading...")
    model_dir = model.download()
    print("✅ Model downloaded to:", model_dir)
except Exception as e:
    print("❌ Error fetching model:", e)
