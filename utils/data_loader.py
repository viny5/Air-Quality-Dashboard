import pandas as pd
import numpy as np
import pickle
import joblib
import streamlit as st
from utils.feature_engineering import add_aqi_features


# ✅ Normalize column names
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


# ✅ Load dataset (with error handling)
@st.cache_data
def load_data(file_path=None):
    try:
        if file_path is None:
            df = pd.read_parquet("cleaned_air_quality_data.parquet")
        else:
            df = pd.read_parquet(file_path) if file_path.endswith(".parquet") else pd.read_csv(file_path)

        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.loc[:, ~df.columns.duplicated()]
        df = normalize_columns(df)

        subindex_renames = {
            "pm2.5_sub_index": "pm25_sub_index",
            "nox_sub_index": "no2_sub_index"
        }
        df.rename(columns=subindex_renames, inplace=True)

        return df

    except FileNotFoundError:
        st.error("❌ Required data file not found. Please ensure it exists in your repo.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Failed to load data: {e}")
        st.stop()


# ✅ Load trained hybrid models
@st.cache_resource
def load_models():
    try:
        with open("trained_hybrid_models.pkl", "rb") as f:
            model_array = pickle.load(f)

        if isinstance(model_array, dict):
            return model_array
        elif isinstance(model_array, np.ndarray) and len(model_array) == 1 and isinstance(model_array[0], dict):
            return model_array[0]
        else:
            raise ValueError("Invalid model structure in trained_hybrid_models.pkl")

    except FileNotFoundError:
        st.error("❌ 'trained_hybrid_models.pkl' not found.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading trained models: {e}")
        st.stop()


# ✅ Load feature set used during training
@st.cache_data
def load_features_used():
    try:
        with open("features_used_in_training.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("❌ 'features_used_in_training.pkl' not found.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading training features: {e}")
        st.stop()


# ✅ Load selected features (RFECV or others)
@st.cache_data
def load_selected_features():
    try:
        with open("rfecv_selected_features_all.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("❌ 'rfecv_selected_features_all.pkl' not found.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading selected features: {e}")
        st.stop()


# ✅ Load AQI model and features
@st.cache_resource
def load_aqi_model_and_features():
    try:
        model = joblib.load("aqi_model.pkl")
        features = joblib.load("aqi_features.pkl")
        return model, features
    except FileNotFoundError:
        st.error("❌ AQI model or features not found.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Failed to load AQI model or features: {e}")
        st.stop()


# ✅ Return fully prepared dataset
@st.cache_data
def get_prepared_data(file_path=None):
    df = load_data(file_path)
    df = add_aqi_features(df)
    df = df[~df.index.duplicated()].sort_index()
    return df


# ✅ Ensure required features exist
def ensure_model_features(df: pd.DataFrame, required_features: list, fill_value=np.nan) -> pd.DataFrame:
    df = df.copy()
    df = df.loc[:, ~df.columns.duplicated()]
    for feat in required_features:
        if feat not in df.columns:
            df[feat] = fill_value
    return df[required_features]
