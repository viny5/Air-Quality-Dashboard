# utils/shap_utils.py

import shap
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from utils.data_loader import load_data, load_features_used

def explain_shap(model, x_input):
    from utils.data_loader import load_data

    # ✅ Normalize column names and patch sub-index names
    def normalize_column(col):
        col = col.lower().strip()
        col = col.replace("pm 2.5", "pm2.5").replace("pm25", "pm2.5")
        col = col.replace("pm 10", "pm10")
        col = col.replace("pm2.5_sub_index", "pm25_sub_index")  # match model
        return col

    x_input = x_input.copy()
    x_input.columns = [normalize_column(c) for c in x_input.columns]
    x_input = x_input.rename(columns={"pm2.5_sub_index": "pm25_sub_index"})  # final patch

    # Background for SHAP (fallback-safe)
    try:
        full_data = load_data()
        full_data.columns = [normalize_column(c) for c in full_data.columns]
        full_data = full_data.rename(columns={"pm2.5_sub_index": "pm25_sub_index"})
        background = full_data[x_input.columns].dropna()
        if len(background) >= 2:
            background_sample = background.sample(min(100, len(background)), random_state=42)
        else:
            background_sample = x_input.copy()
    except Exception as e:
        st.warning(f"⚠️ Could not load background data for SHAP: {e}")
        background_sample = x_input.copy()

    try:
        explainer = shap.Explainer(model, background_sample)
        shap_values = explainer(x_input)

        shap_df = pd.DataFrame({
            "Feature": x_input.columns,
            "Feature Value": x_input.iloc[0].values,
            "SHAP Value": shap_values.values[0]
        }).sort_values(by="SHAP Value", key=abs, ascending=False)

        return shap_df, shap_values
    except Exception as e:
        st.warning(f"⚠️ SHAP explanation failed: {e}")
        return None, None

def show_shap_bar(x_input, shap_values):
    shap_df = pd.DataFrame({
        "Feature": x_input.columns,
        "SHAP Value": shap_values.values[0]
    }).sort_values(by="SHAP Value", key=abs, ascending=False)

    st.subheader("🔍 SHAP Top Features")
    fig, ax = plt.subplots()
    sns.barplot(x="SHAP Value", y="Feature", data=shap_df.head(10), ax=ax)
    st.pyplot(fig)


def show_shap_summary_bar(X, shap_values, max_display=10):
    """
    Displays a SHAP summary bar plot for multiple rows using average absolute SHAP values.

    Parameters:
    - X: DataFrame of input features.
    - shap_values: SHAP values (matching X).
    - max_display: maximum number of top features to show.
    """
    shap_abs = np.abs(shap_values.values).mean(axis=0)
    top_indices = np.argsort(shap_abs)[-max_display:][::-1]
    top_features = X.columns[top_indices]
    top_values = shap_abs[top_indices]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(top_features[::-1], top_values[::-1], color='steelblue')
    ax.set_xlabel("Average SHAP Value (Impact on Model Output)")
    ax.set_title("Top SHAP Feature Contributions (Last 5 Days)")
    st.pyplot(fig)
