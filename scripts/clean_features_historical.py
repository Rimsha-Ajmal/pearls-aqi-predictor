# clean_features_historical.py

import pandas as pd
import numpy as np

# ---------------- LOAD FEATURES ----------------
input_path = "computed_features_historical_selected.csv"  # updated input
df = pd.read_csv(input_path)
print(f"✅ Loaded feature dataset: {df.shape}")

# ---------------- CLEANING ----------------
exclude_cols = ['datetime']
numeric_cols = [c for c in df.columns if c not in exclude_cols]

# Replace impossible zeros with NaN (except aqi and wind_speed)
for col in numeric_cols:
    if col not in ['aqi', 'wind_speed'] and not col.startswith('aqi_t_plus'):
        df[col] = df[col].replace(0, np.nan)

# Forward-fill then backward-fill for missing values
df = df.ffill().bfill()

# Replace any remaining NaNs with column median
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# ---------------- VERIFY ----------------
remaining_nans = df.isna().sum().sum()
remaining_zeros = (df[numeric_cols] == 0).sum().sum()
print(f"🔍 Remaining NaN values: {remaining_nans}")
print(f"🔍 Remaining zeros in numeric columns: {remaining_zeros}")

# ---------------- SAVE CLEAN DATA ----------------
output_path = "computed_features_historical_selected_cleaned.csv"
df.to_csv(output_path, index=False)
print(f"✅ Final cleaned dataset saved ({df.shape[0]} rows, {df.shape[1]} cols)")
