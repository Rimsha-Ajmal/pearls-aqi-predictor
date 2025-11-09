# Pearls AQI Predictor

**Overview:**  
Predict the Air Quality Index (AQI) for the next 3 days using real-time and historical weather & pollutant data. The project uses a serverless ML pipeline with automated data fetching, feature engineering, model training, and live predictions via a web dashboard.

**Technology Stack:**  
- Python, Scikit-learn, TensorFlow  
- Hopsworks Feature Store  
- GitHub Actions for CI/CD  
- Streamlit for dashboard  
- SHAP for feature importance  
- OpenWeatherMap & Open-Meteo APIs  

**Project Flow:**  
1. Fetch **real-time** and **historical** data; store in Hopsworks feature groups.  
2. Compute features: lag, rolling, time-based, and future targets.  
3. Clean dataset (impute missing values, remove impossible zeros).  
4. Train ML models (Random Forest, Ridge, GradientBoosting).  
5. Evaluate models using RMSE, MAE, R²; select best model.  
6. Use SHAP to identify top impactful features; retrain model on top features.  
7. CI/CD Pipelines:  
   - Hourly data fetch → raw_observations  
   - Daily retraining → update model if improved  
8. Web dashboard displays:  
   - Real-time data & graphs  
   - Model training results & comparison  
   - Feature importance & insights  
   - AQI predictions for next 3 days with alerts  

**Future Improvements:**  
- Replace AQI 1–5 scale with exact numeric values.  
- Explore advanced/deep learning models for better forecasting.

### Project Report

You can download the project report here:

[📄 Pearls AQI Predictor Report (Word)](Pearls_AQI_Predictor_Report.docx)

### How to Run the Project

Follow these steps to set up and run the project locally:

1. **Create and activate a virtual environment**
python -m venv test_env
test_env\Scripts\activate      # On Windows
# or
source test_env/bin/activate   # On macOS/Linux

2. **Install the dependencies**
pip install -r requirements.txt

3. **Run the Streamlit app**
streamlit run app.py


