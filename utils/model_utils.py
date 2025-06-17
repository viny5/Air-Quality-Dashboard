# utils/model_utils.py

import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
import statsmodels.api as sm
import streamlit as st

def train_classification_model(df):
    # Clip extreme values before training
    df['co_sub_index'] = df['co_sub_index'].clip(0, 500)

    features = ['pm2.5', 'pm10', 'so2', 'no2', 'co', 'o3']
    df = df.dropna(subset=features + ['aqi_category']).copy()

    # Use AQI categories as labels
    X = df[features]
    y = df['aqi_category']  # Use AQI category as target

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    return clf, features

def classification_prediction(df, clf, features):
    latest = df[features].dropna().tail(1)
    if latest.empty:
        return None, None
    pred = clf.predict(latest)[0]
    return pred, latest

def forecast_all_pollutants(df, model_dict, feature_dict):
    """
    Forecasts all pollutants and AQI using trained models.
    Automatically remaps 'pm2.5_sub_index' → 'pm25_sub_index' to match model expectations.
    """
    df = df.copy()
    df = df.rename(columns={"pm2.5_sub_index": "pm25_sub_index"})

    forecasts = {}
    for pollutant_raw, model in model_dict.items():
        # Normalize the pollutant name to lowercase to avoid case mismatch
        pollutant = pollutant_raw.lower()

        # Fetch the features for the pollutant and normalize them
        feats_raw = feature_dict.get(pollutant, [])

        # Normalize feature names (e.g., "pm25" -> "pm2.5")
        feats = [
            f.lower().replace("pm25", "pm2.5").replace("pm 2.5", "pm2.5").strip()
            for f in feats_raw
        ]
        feats = [f if f != "pm2.5_sub_index" else "pm25_sub_index" for f in feats]

        # Ensure all expected features are present in the DataFrame
        for f in feats:
            if f not in df.columns:
                df[f] = np.nan  # Add the missing column with NaN values

        # Align and reorder features as expected by the model
        latest_input = df[feats].tail(1)

        # If no features or the latest input is empty, skip this pollutant
        if not feats or df[feats].dropna().empty:
            forecasts[pollutant_raw] = None
            continue

        latest_input = df[feats].dropna().tail(1)
        if latest_input.empty:
            forecasts[pollutant_raw] = None
        else:
            try:
                # Predict the pollutant level
                pred = float(model.predict(latest_input)[0])
                forecasts[pollutant_raw] = round(pred, 2)
            except Exception:
                forecasts[pollutant_raw] = None

    return forecasts

def arima_forecast(df, target_col, forecast_steps=7):
    try:
        series = df[target_col].dropna()
        model = sm.tsa.ARIMA(series, order=(3, 1, 2))
        results = model.fit()
        forecast = results.forecast(steps=24 * forecast_steps)
        forecast.index = pd.date_range(start=series.index[-1] + pd.Timedelta(hours=1), periods=len(forecast), freq='H')
        return forecast
    except Exception as e:
        st.error(f"ARIMA forecast failed: {e}")
        return None

def summarize_forecast_with_huggingface(text):
    try:
        API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
        response = requests.post(API_URL, json={"inputs": text})
        summary = response.json()[0]["summary_text"]
        return summary
    except Exception as e:
        return f"⚠️ Hugging Face summarization failed: {e}"

def safe_date_range(df):
    try:
        if df.empty or df.index.min() is pd.NaT or df.index.max() is pd.NaT:
            today = pd.Timestamp.today().date()
            return [today, today]
        return [df.index.min().date(), df.index.max().date()]
    except Exception:
        today = pd.Timestamp.today().date()
        return [today, today]

def hybrid_feature_selection(X, y, top_n=20):
    models = {
        "rf": RandomForestRegressor(n_estimators=100),
        "et": ExtraTreesRegressor(n_estimators=100),
        "xgb": XGBRegressor(n_estimators=100, verbosity=0)
    }

    importances = {}
    for name, model in models.items():
        model.fit(X, y)
        importances[name] = model.feature_importances_

    importances_df = pd.DataFrame(importances, index=X.columns)
    importances_df["mean"] = importances_df.mean(axis=1)
    top_feats = importances_df["mean"].sort_values(ascending=False).head(top_n).index.tolist()
    return top_feats