from utils.hops import connect_hopsworks

project, _ = connect_hopsworks()
mr = project.get_model_registry()
model = mr.get_model("AQI_RandomForest_H72", version=None)
model_dir = model.download()

import os
print("Model directory:", model_dir)
print("Files:", os.listdir(model_dir))
