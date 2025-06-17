import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data
def add_aqi_features(df, include_lag_features=True, include_rolling_features=True):
    df = df.copy()

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # Remove legacy capitalized sub_index columns
    df = df.drop(columns=[col for col in df.columns if "_sub_index" in col and not col.islower()], errors="ignore")

    # --- Time Features ---
    if df.index.name is not None and isinstance(df.index, pd.DatetimeIndex):
        df["hour"] = df.index.hour
        df["day_of_week"] = df.index.dayofweek
        df["month"] = df.index.month
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # --- Pollutant Features ---
    pollutant_cols = ["pm2.5", "pm10", "so2", "no2", "co", "o3"]

    if include_lag_features:
        for col in pollutant_cols:
            if col in df.columns:
                df[f"{col}_lag_1"] = df[col].shift(1)
                df[f"{col}_lag_24"] = df[col].shift(24)

    if include_rolling_features:
        for col in pollutant_cols:
            if col in df.columns:
                df[f"{col}_roll_mean_3"] = df[col].rolling(3).mean()
                df[f"{col}_roll_mean_24"] = df[col].rolling(24).mean()

    # --- Optional CO to PPM ---
    if "co" in df.columns:
        df["co_ppm"] = df["co"].apply(lambda x: x / 1.145 if pd.notnull(x) else np.nan)

    # --- EPA Sub-Index Calculation ---
    def safe_apply(func, series):
        return series.apply(lambda x: func(x) if pd.notnull(x) else np.nan)

    def get_pm25_subindex(x):
        if x <= 12: return x * 50 / 12
        elif x <= 35.4: return 50 + (x - 12) * 50 / (35.4 - 12)
        elif x <= 55.4: return 100 + (x - 35.5) * 50 / (55.4 - 35.5)
        elif x <= 150.4: return 150 + (x - 55.5) * 100 / (150.4 - 55.5)
        elif x <= 250.4: return 200 + (x - 150.5) * 100 / (250.4 - 150.5)
        elif x <= 350.4: return 300 + (x - 250.5) * 100 / (350.4 - 250.5)
        else: return 400 + (x - 350.5) * 100 / (500.4 - 350.5)

    def get_pm10_subindex(x):
        if x <= 54: return x * 50 / 54
        elif x <= 154: return 50 + (x - 55) * 50 / (154 - 55)
        elif x <= 254: return 100 + (x - 155) * 50 / (254 - 155)
        elif x <= 354: return 150 + (x - 255) * 100 / (354 - 255)
        elif x <= 424: return 200 + (x - 355) * 100 / (424 - 355)
        elif x <= 504: return 300 + (x - 425) * 100 / (504 - 425)
        else: return 400 + (x - 505) * 100 / (604 - 505)

    def get_so2_subindex(x):
        if x <= 35: return x * 50 / 35
        elif x <= 75: return 50 + (x - 36) * 50 / (75 - 36)
        elif x <= 185: return 100 + (x - 76) * 50 / (185 - 76)
        elif x <= 304: return 150 + (x - 186) * 100 / (304 - 186)
        elif x <= 604: return 200 + (x - 305) * 100 / (604 - 305)
        else: return 300 + (x - 605) * 100 / (1004 - 605)

    def get_no2_subindex(x):
        if x <= 53: return x * 50 / 53
        elif x <= 100: return 50 + (x - 54) * 50 / (100 - 54)
        elif x <= 360: return 100 + (x - 101) * 50 / (360 - 101)
        elif x <= 649: return 150 + (x - 361) * 100 / (649 - 361)
        elif x <= 1249: return 200 + (x - 650) * 100 / (1249 - 650)
        else: return 300 + (x - 1250) * 100 / (2049 - 1250)

    def get_co_subindex(x):
        if x <= 4.4: return x * 50 / 4.4
        elif x <= 9.4: return 50 + (x - 4.5) * 50 / (9.4 - 4.5)
        elif x <= 12.4: return 100 + (x - 9.5) * 50 / (12.4 - 9.5)
        elif x <= 15.4: return 150 + (x - 12.5) * 100 / (15.4 - 12.5)
        elif x <= 30.4: return 200 + (x - 15.5) * 100 / (30.4 - 15.5)
        elif x <= 40.4: return 300 + (x - 30.5) * 100 / (40.4 - 30.5)
        else: return 400 + (x - 40.5) * 100 / (50.4 - 40.5)

    def get_o3_subindex(x):
        if x <= 0.054: return x * 50 / 0.054
        elif x <= 0.070: return 50 + (x - 0.055) * 50 / (0.070 - 0.055)
        elif x <= 0.085: return 100 + (x - 0.071) * 50 / (0.085 - 0.071)
        elif x <= 0.105: return 150 + (x - 0.086) * 100 / (0.105 - 0.086)
        elif x <= 0.200: return 200 + (x - 0.106) * 100 / (0.200 - 0.106)
        else: return 300 + (x - 0.201) * 100 / (0.404 - 0.201)

    # --- Apply Sub-Index Calculations ---
    if "pm2.5" in df.columns:
        df["pm2.5_sub_index"] = safe_apply(get_pm25_subindex, df["pm2.5"])
    if "pm10" in df.columns:
        df["pm10_sub_index"] = safe_apply(get_pm10_subindex, df["pm10"])
    if "so2" in df.columns:
        df["so2_sub_index"] = safe_apply(get_so2_subindex, df["so2"])
    if "no2" in df.columns:
        df["no2_sub_index"] = safe_apply(get_no2_subindex, df["no2"])
    if "co" in df.columns:
        df["co_sub_index"] = safe_apply(get_co_subindex, df["co"])
    if "o3" in df.columns:
        df["o3_sub_index"] = safe_apply(get_o3_subindex, df["o3"])

    # ✅ Clip sub-index values to 0–500
    for col in ["pm2.5_sub_index", "pm10_sub_index", "so2_sub_index", "no2_sub_index", "co_sub_index", "o3_sub_index"]:
        if col in df.columns:
            df[col] = df[col].clip(0, 500)

    # ✅ Fill NaNs in sub-index columns with mean (safe method)
    subindex_cols = [col for col in df.columns if col.endswith("_sub_index")]
    for col in subindex_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].mean())

    # ✅ Remove fully NaN sub-index columns (if still any)
    df = df.drop(columns=[col for col in subindex_cols if df[col].isna().all()], errors="ignore")

    # ✅ Compute AQI
    subindex_cols = [col for col in df.columns if col.endswith("_sub_index")]
    if subindex_cols:
        df["aqi"] = df[subindex_cols].max(axis=1)
    else:
        df["aqi"] = np.nan

    # ✅ AQI Category
    def categorize_aqi(aqi):
        if pd.isna(aqi): return np.nan
        elif aqi <= 50: return "Good"
        elif aqi <= 100: return "Moderate"
        elif aqi <= 150: return "Unhealthy for Sensitive Groups"
        elif aqi <= 200: return "Unhealthy"
        elif aqi <= 300: return "Very Unhealthy"
        else: return "Hazardous"

    df["aqi_category"] = df["aqi"].apply(categorize_aqi)

    return df