import hopsworks
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

PROJECT_NAME = os.getenv("PROJECT_NAME")
API_KEY = os.getenv("HOPSWORKS_API_KEY")

try:
    # Login to Hopsworks
    project = hopsworks.login(project=PROJECT_NAME, api_key_value=API_KEY)
    print(f"✅ Connected to Hopsworks project: {project.name}")

    # Connect to Feature Store
    fs = project.get_feature_store()
    print("✅ Feature Store connected:", fs.name)

    # Since your API requires explicit names:
    feature_groups_to_check = ["model_features", "raw_observations"]
    found_feature_groups = []

    for fg_name in feature_groups_to_check:
        try:
            fg = fs.get_feature_group(name=fg_name, version=1)
            found_feature_groups.append(fg_name)
        except Exception:
            pass

    print(f"📂 Feature Groups Found: {found_feature_groups}")

    # Connect to Model Registry
    mr = project.get_model_registry()

    # Your trained model name
    model_name = "RandomForest_H72"

    try:
        model = mr.get_model(model_name, version=1)
        print(f"🤖 Model Found: {model_name}")
    except Exception:
        print(f"⚠️ Model '{model_name}' not found! Check registry.")

    print("✅ Hopsworks connection + services verified")

except Exception as e:
    print("❌ Connection failed")
    print(e)
