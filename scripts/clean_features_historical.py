# ==========================================
# 🧹 Clean AQI Feature Dataset (Remove Zero & Missing Values)
# ==========================================

import pandas as pd
import numpy as np
import os

# ---------------- CONFIG ----------------
INPUT_FILE = "model_features.csv"
OUTPUT_FILE = "model_features_v1_final.csv"

# ---------------- LOAD FEATURES ----------------
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"❌ '{INPUT_FILE}' not found. Please place it in the same directory as this script.")

df = pd.read_csv(INPUT_FILE)
print(f"✅ Loaded feature dataset: {df.shape}")

# ---------------- ZERO VALUE ANALYSIS ----------------
exclude_cols = ['datetime']
numeric_cols = [c for c in df.columns if c not in exclude_cols]

results = []
for col in numeric_cols:
    zero_count = (df[col] == 0).sum()
    zero_percent = (zero_count / len(df)) * 100
    try:
        corr_with_aqi = df[col].corr(df['aqi'])
    except Exception:
        corr_with_aqi = np.nan
    results.append({
        'column': col,
        'zero_count': zero_count,
        'zero_percent': round(zero_percent, 2),
        'corr_with_aqi': round(corr_with_aqi, 3)
    })

zero_analysis = pd.DataFrame(results).sort_values(by='zero_percent', ascending=False)
print("\n📊 Zero Value Analysis (Top Columns):")
print(zero_analysis.head(10))

# ---------------- REMOVE HIGH ZERO FEATURES ----------------
cols_to_drop = zero_analysis[zero_analysis['zero_percent'] > 80]['column'].tolist()
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
print(f"\n🧹 Dropped {len(cols_to_drop)} mostly-zero columns: {cols_to_drop}")

# ---------------- HANDLE ZERO & MISSING VALUES ----------------
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].replace(0, np.nan)

# Recompute month_sin if 'month' column exists
if "month" in df.columns:
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)

# Fill missing values using forward + backward fill
df = df.ffill().bfill()

print(f"🔍 Remaining missing values: {df.isna().sum().sum()}")

# ---------------- SAVE CLEAN DATA ----------------
df.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ Final cleaned dataset saved: '{OUTPUT_FILE}'")
print(f"📏 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
