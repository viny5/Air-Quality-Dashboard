# data_loader.py

import pandas as pd
import numpy as np
import pickle
import joblib
import streamlit as st
from utils.feature_engineering import add_aqi_features  # Make sure this exists

# ✅ Normalize column names (standardized casing + underscores)
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_", regex=False)
        .str.replace("pm25", "pm2.5", case=False, regex=False)
        .str.replace("pm_2.5", "pm2.5", case=False, regex=False)
        .str.replace("pm_10", "pm10", case=False, regex=False)
        .str.replace("nox", "no2", case=False, regex=False)
        .str.replace("PM2.5", "pm2.5", case=False, regex=False)
        .str.replace("PM10", "pm10", case=False, regex=False)
        .str.replace("CO", "co", case=False, regex=False)
        .str.replace("SO2", "so2", case=False, regex=False)
        .str.replace("NO2", "no2", case=False, regex=False)
        .str.replace("O3", "o3", case=False, regex=False)
        .str.replace("TEMP", "temp", case=False, regex=False)
        .str.replace("PRES", "pres", case=False, regex=False)
        .str.replace("DEWP", "dewp", case=False, regex=False)
        .str.replace("RAIN", "rain", case=False, regex=False)
        .str.replace("WSPM", "wspm", case=False, regex=False)
        .str.replace("AQI", "aqi", case=False, regex=False)
        .str.replace("AQI_Category", "aqi_category", case=False, regex=False)
        .str.replace("AQI_Color", "aqi_color", case=False, regex=False)
        .str.lower()
    )
    return df

# ✅ Load dataset (from default or user-uploaded path)
def load_data(file_path=None):
    if file_path is None:
        df = pd.read_parquet("cleaned_air_quality_data.parquet")
    else:
        df = pd.read_parquet(file_path) if file_path.endswith(".parquet") else pd.read_csv(file_path)

    df.index = pd.to_datetime(df.index)
    df = df.loc[:, ~df.columns.duplicated()]
    df = normalize_columns(df)

    # ✅ Normalize any sub-index naming to match trained model
    subindex_renames = {
        "pm2.5_sub_index": "pm25_sub_index",
        "nox_sub_index": "no2_sub_index"
    }
    df.rename(columns=subindex_renames, inplace=True)

    return df

# ✅ Load trained hybrid models
@st.cache_resource
def load_models():
    with open("trained_hybrid_models.pkl", "rb") as f:
        model_array = pickle.load(f)

    if isinstance(model_array, dict):
        return model_array

    elif isinstance(model_array, np.ndarray):
        if len(model_array) == 1 and isinstance(model_array[0], dict):
            return model_array[0]
        else:
            st.error("❌ Invalid format in trained_hybrid_models.pkl — expected a dict or single-item array.")
            return {}

    else:
        st.error(f"❌ Unexpected model format: {type(model_array)}")
        return {}

# ✅ Load feature sets used during training
@st.cache_data
def load_features_used():
    with open("features_used_in_training.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_selected_features():
    with open("rfecv_selected_features_all.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_aqi_model_and_features():
    try:
        model = joblib.load("aqi_model.pkl")
        features = joblib.load("aqi_features.pkl")
        return model, features
    except Exception as e:
        st.error(f"❌ Failed to load AQI model or features: {e}")
        return None, []

# ✅ Return fully prepared dataset (normalized, deduplicated, engineered)
@st.cache_data
def get_prepared_data(file_path=None):
    df = load_data(file_path)

    # ✅ Always apply feature engineering
    df = add_aqi_features(df)

    df = df[~df.index.duplicated()].sort_index()
    return df

# ✅ Ensure required features are present and in the correct order
def ensure_model_features(df: pd.DataFrame, required_features: list, fill_value=np.nan) -> pd.DataFrame:
    df = df.copy()
    df = df.loc[:, ~df.columns.duplicated()]

    for feat in required_features:
        if feat not in df.columns:
            df[feat] = fill_value

    # ✅ Return features in exact expected order
    return df[required_features]