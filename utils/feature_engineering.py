import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data
def add_aqi_features(df, include_lag_features=True, include_rolling_features=True):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    # Drop old sub_index columns (in case of legacy uppercase ones)
    df.drop(columns=[col for col in df.columns if "_sub_index" in col and not col.islower()], errors="ignore", inplace=True)

    # --- Time Features ---
    if isinstance(df.index, pd.DatetimeIndex):
        df["hour"] = df.index.hour
        df["day_of_week"] = df.index.dayofweek
        df["month"] = df.index.month
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # --- Pollutant-based Features ---
    pollutants = ["pm2.5", "pm10", "so2", "no2", "co", "o3"]

    if include_lag_features:
        for col in pollutants:
            if col in df.columns:
                df[f"{col}_lag_1"] = df[col].shift(1)
                df[f"{col}_lag_24"] = df[col].shift(24)

    if include_rolling_features:
        for col in pollutants:
            if col in df.columns:
                df[f"{col}_roll_mean_3"] = df[col].rolling(3).mean()
                df[f"{col}_roll_mean_24"] = df[col].rolling(24).mean()

    # --- Convert CO to ppm ---
    if "co" in df.columns:
        df["co_ppm"] = df["co"].apply(lambda x: x / 1.145 if pd.notnull(x) else np.nan)

    # --- Sub-Index Calculation Helpers ---
    def safe_apply(func, series):
        return series.apply(lambda x: func(x) if pd.notnull(x) else np.nan)

    # Sub-index functions (EPA standards)
    def get_pm25_subindex(x):
        bounds = [(0, 12, 0, 50), (12.1, 35.4, 50, 100), (35.5, 55.4, 100, 150),
                  (55.5, 150.4, 150, 200), (150.5, 250.4, 200, 300),
                  (250.5, 350.4, 300, 400), (350.5, 500.4, 400, 500)]
        for low, high, si_low, si_high in bounds:
            if x <= high:
                return si_low + (x - low) * (si_high - si_low) / (high - low)
        return np.nan

    def get_pm10_subindex(x):
        bounds = [(0, 54, 0, 50), (55, 154, 50, 100), (155, 254, 100, 150),
                  (255, 354, 150, 200), (355, 424, 200, 300),
                  (425, 504, 300, 400), (505, 604, 400, 500)]
        for low, high, si_low, si_high in bounds:
            if x <= high:
                return si_low + (x - low) * (si_high - si_low) / (high - low)
        return np.nan

    def get_so2_subindex(x):
        bounds = [(0, 35, 0, 50), (36, 75, 50, 100), (76, 185, 100, 150),
                  (186, 304, 150, 200), (305, 604, 200, 300), (605, 1004, 300, 400)]
        for low, high, si_low, si_high in bounds:
            if x <= high:
                return si_low + (x - low) * (si_high - si_low) / (high - low)
        return np.nan

    def get_no2_subindex(x):
        bounds = [(0, 53, 0, 50), (54, 100, 50, 100), (101, 360, 100, 150),
                  (361, 649, 150, 200), (650, 1249, 200, 300), (1250, 2049, 300, 400)]
        for low, high, si_low, si_high in bounds:
            if x <= high:
                return si_low + (x - low) * (si_high - si_low) / (high - low)
        return np.nan

    def get_co_subindex(x):
        bounds = [(0, 4.4, 0, 50), (4.5, 9.4, 50, 100), (9.5, 12.4, 100, 150),
                  (12.5, 15.4, 150, 200), (15.5, 30.4, 200, 300),
                  (30.5, 40.4, 300, 400), (40.5, 50.4, 400, 500)]
        for low, high, si_low, si_high in bounds:
            if x <= high:
                return si_low + (x - low) * (si_high - si_low) / (high - low)
        return np.nan

    def get_o3_subindex(x):
        bounds = [(0, 0.054, 0, 50), (0.055, 0.070, 50, 100), (0.071, 0.085, 100, 150),
                  (0.086, 0.105, 150, 200), (0.106, 0.200, 200, 300),
                  (0.201, 0.404, 300, 400)]
        for low, high, si_low, si_high in bounds:
            if x <= high:
                return si_low + (x - low) * (si_high - si_low) / (high - low)
        return np.nan

    # Apply sub-index calculations
    subindex_map = {
        "pm2.5": get_pm25_subindex,
        "pm10": get_pm10_subindex,
        "so2": get_so2_subindex,
        "no2": get_no2_subindex,
        "co": get_co_subindex,
        "o3": get_o3_subindex
    }

    for pol, func in subindex_map.items():
        if pol in df.columns:
            df[f"{pol}_sub_index"] = safe_apply(func, df[pol]).clip(0, 500)

    # --- Final AQI & Category ---
    subindex_cols = [col for col in df.columns if col.endswith("_sub_index")]
    df["aqi"] = df[subindex_cols].max(axis=1)

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