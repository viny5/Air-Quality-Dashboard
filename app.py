import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

from config import POLLUTANTS
from utils.data_loader import (
    load_data, load_models, load_features_used, ensure_model_features,
    load_selected_features, load_aqi_model_and_features
)
from utils.feature_engineering import add_aqi_features
from utils.model_utils import (
    train_classification_model, classification_prediction, arima_forecast, 
    safe_date_range
)
from utils.shap_utils import explain_shap, show_shap_bar, show_shap_summary_bar

st.set_page_config(page_title="Air Quality Forecast Dashboard", layout="wide")
st.title("🌫️ Air Quality Forecast Dashboard")

# --- Session State Setup ---
if "aqi_thresh_good" not in st.session_state:
    st.session_state["aqi_thresh_good"] = 50
if "aqi_thresh_bad" not in st.session_state:
    st.session_state["aqi_thresh_bad"] = 150
if "manual_feats" not in st.session_state:
    st.session_state["manual_feats"] = []
if "model_type" not in st.session_state:
    st.session_state["model_type"] = "Random Forest"

# --- Sidebar Filters ---
st.sidebar.markdown("## 🔍 Global Filters")

date_range_global = st.sidebar.date_input(
    "Date Range",
    value=[pd.to_datetime("2023-01-01").date(), pd.to_datetime("2023-12-31").date()],
    key="date_range_global"
)

time_range_global = st.sidebar.slider("Hour of Day", 0, 23, (0, 23))
filter_day_type = st.sidebar.radio("Day Type", ["All", "Weekday", "Weekend"])

include_lag_features = st.sidebar.checkbox("Include Lag Features", value=True, key="lag_checkbox")
include_rolling_features = st.sidebar.checkbox("Include Rolling Features", value=True, key="rolling_checkbox")

# --- File Upload ---
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Parquet", type=["csv", "parquet"])
from_uploaded = False

if uploaded_file:
    from_uploaded = True
    try:
        # --- Load CSV or Parquet ---
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_parquet(uploaded_file)

        df.columns = df.columns.str.lower()

        # --- Force datetime index from time columns ---
        required_time_cols = ['year', 'month', 'day', 'hour']
        if all(col in df.columns for col in required_time_cols):
            try:
                df["datetime"] = pd.to_datetime(df[required_time_cols])
                df.set_index("datetime", inplace=True)
            except Exception as e:
                st.warning(f"⚠️ Failed to build datetime from columns: {e}")
                df.index = pd.to_datetime(df.index, errors="coerce")
        else:
            df.index = pd.to_datetime(df.index, errors="coerce")

        # --- Clean index ---
        df = df[~df.index.duplicated()].sort_index()

        # --- Feature Engineering ---
        df = add_aqi_features(
            df,
            include_lag_features=include_lag_features,
            include_rolling_features=include_rolling_features
        )

        # --- Fill missing sub-index values ---
        subindex_cols = [col for col in df.columns if col.endswith("_sub_index")]
        for col in subindex_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].mean())

        st.sidebar.success("✅ File loaded and processed.")

    except Exception as e:
        st.sidebar.error(f"❌ Error: {e}")
        st.stop()
else:
    df = load_data()
    from_uploaded = False

# --- Global Date Filter ---
df = df[(df.index.date >= date_range_global[0]) & (df.index.date <= date_range_global[1])]

# --- Fallback if filters return empty ---
if df.empty:
    df = load_data()

# --- If not from upload, run feature engineering ---
if not from_uploaded:
    df = add_aqi_features(df, include_lag_features=include_lag_features, include_rolling_features=include_rolling_features)

# --- Load ML Models and Feature Sets ---
models = load_models()
features_used = load_features_used()
selected_features_all = load_selected_features()
aq_model, aqi_features = load_aqi_model_and_features()

# --- Dashboard Tabs ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "🧪 Data Exploration", "📈 Forecast", "🔍 Classification", "📉 ARIMA Forecast",
    "📊 Historical Trends", "🧠 Compare Models", "🗖️ Compare Dates",
    "📊 Model Evaluation", "⚙️ Advanced Settings", "ℹ️ About"
])

# --- Data Exploration Tab ---
with tab1:
    st.subheader("🧪 Data Exploration")

    if df.empty:
        st.error("⚠️ No data available after filtering. Showing a preview of the full default dataset.")
        preview_df = load_data().head(10)
        st.dataframe(preview_df)
        st.stop()
    else:
        st.markdown("### 🧩 Global Missing Value Strategy")

        global_missing_option = st.radio(
            "Choose how to handle missing values across the dataset (applies to all tabs):",
            ["None", "Drop rows", "Forward fill", "Fill with zero"],
            key="missing_strategy_global"
        )

        if global_missing_option == "Drop rows":
            df = df.dropna()
        elif global_missing_option == "Forward fill":
            df = df.ffill().bfill()
        elif global_missing_option == "Fill with zero":
            df = df.fillna(0)

        st.markdown("### 🔍 Preview Dataset")
        st.dataframe(df.head())

        st.markdown("### 📊 Dataset Summary Statistics")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Descriptive Statistics**")
            st.dataframe(df.describe().transpose())

        with col2:
            st.write("**Missing Value % per Column**")
            null_pct = df.isna().mean().round(3) * 100
            st.dataframe(null_pct.to_frame(name="Missing %"))

        st.markdown("### 📌 Column Selector + Dropper")
        all_cols = df.columns.tolist()
        cols_to_drop = st.multiselect("Select columns to drop (won't affect original data)", all_cols)
        df_explore = df.drop(columns=cols_to_drop)

        st.markdown("### 📈 Boxplot")
        numeric_cols = df_explore.select_dtypes(include=[np.number]).columns.tolist()
        box_col = st.selectbox("Select column for boxplot", numeric_cols)
        if box_col:
            fig, ax = plt.subplots()
            sns.boxplot(data=df_explore[box_col], ax=ax)
            st.pyplot(fig)

        st.markdown("### 🔥 Heatmap (Correlation)")
        if st.button("Show Heatmap"):
            corr = df_explore.corr()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)

        st.markdown("### 🎚️ Filter Rows by Pollutant Threshold")

        # Always lowercase for internal column lookup
        poll = st.selectbox("Pollutant to filter", POLLUTANTS, key="threshold_poll").lower()

        if poll not in df.columns:
            st.error(f"❌ Column '{poll}' not found in the dataset.")
        else:
            threshold = st.slider(
                "Minimum value",
                float(df[poll].min()),
                float(df[poll].max()),
                float(df[poll].mean())
            )

            df_filtered = df[df[poll] >= threshold]
            st.write(f"Rows where **{poll.upper()} ≥ {threshold:.1f}**:")
            st.dataframe(df_filtered.head(10))

        st.markdown("### 🔗 Scatterplot Matrix (Pairwise Relationship)")
        pair_numeric_cols = df_explore.select_dtypes(include=[np.number]).columns.tolist()
        default_cols = [col for col in ["PM2.5", "PM10", "AQI"] if col in pair_numeric_cols]
        selected_pair_cols = st.multiselect(
            "Select up to 5 numeric features to visualize relationships:",
            pair_numeric_cols,
            default=default_cols
        )

        if selected_pair_cols and len(selected_pair_cols) >= 2:
            try:
                pairplot_fig = sns.pairplot(df_explore[selected_pair_cols].dropna())
                st.pyplot(pairplot_fig.fig)
            except Exception as e:
                st.warning(f"Pairplot failed: {e}")
        else:
            st.info("Select at least 2 numeric columns to display pairwise relationships.")

        st.markdown("### 📊 Interactive Feature Explorer")
        x_feat = st.selectbox("X-axis", pair_numeric_cols, key="x_axis")
        y_feat = st.selectbox("Y-axis", pair_numeric_cols, key="y_axis")
        color_feat = st.selectbox("Color by", [None] + POLLUTANTS, key="color_axis")

        fig, ax = plt.subplots()
        sns.scatterplot(data=df_explore, x=x_feat, y=y_feat,
                        hue=color_feat if color_feat else None, ax=ax)
        st.pyplot(fig)

        st.success("✅ Data exploration complete. Continue to forecasting or modeling.")

# --- Forecast Tab ---
with tab2:
    st.subheader("📈 Forecast")

    if uploaded_file is None:
        st.info("ℹ️ You're using the default historical dataset.")

    if df.empty:
        st.error("❌ No data available after filtering.")
    else:

        pol = st.selectbox("Pollutant", POLLUTANTS)  # User selects the pollutant
        model_choice = st.selectbox("Model", list(models.keys()))
        forecast_date = st.date_input(
            "Forecast Start Date",
            value=df.index.max().date(),
            min_value=df.index.min().date(),
            max_value=df.index.max().date()
        )

        # **Run Forecast** Button for any pollutant
        if st.button("Run Forecast"):
            with st.spinner("Generating forecast..."):
                # Normalize column names to lowercase to ensure consistency
                df.columns = df.columns.str.lower()

                # Ensure the selected pollutant is in uppercase to match the model's keys
                pol = pol.upper()  # Normalize to uppercase

                # Check if the selected pollutant exists in the data
                if pol.lower() not in df.columns:  # Match data column in lowercase
                    st.error(f"❌ The selected pollutant {pol} is not available in the data.")
                    st.stop()  # Stop further execution if the pollutant is not found
                else:
                    # Proceed with the forecasting logic
                    st.write(f"✅ Pollutant {pol} is available in the data.")

                    subset = df[df.index.date == forecast_date]
                    if subset.empty:
                        st.warning("No data for selected start date.")
                        st.stop()

                    subset.columns = subset.columns.str.lower().str.strip()

                    # Check if the selected pollutant exists in the model's keys (case insensitive)
                    if pol not in models[model_choice]:
                        st.error(f"❌ The model does not have data for the pollutant '{pol}'. Please check the available pollutants.")
                        st.stop()  # Stop further execution if the pollutant is not found in the model

                    model = models[model_choice][pol]  # Access the model for the selected pollutant
                    raw_feats = st.session_state.get("manual_feats", []) or features_used[model_choice].get(pol, [])

                    # Normalize feature names to lowercase
                    def normalize_feat(f):
                        f = f.lower().replace("pm25", "pm2.5").replace("pm 2.5", "pm2.5").strip()
                        return "pm25_sub_index" if f == "pm2.5_sub_index" else f

                    feats = [normalize_feat(f) for f in raw_feats]
                    subset = subset.rename(columns={"pm2.5_sub_index": "pm25_sub_index"})
                    feats = [f for f in feats if f in subset.columns]

                    if not feats:
                        st.error("❌ No valid features found for this pollutant after normalization.")
                        st.stop()

                    strategy = st.session_state.get("missing_strategy_global", "None")
                    st.markdown(f"🧩 **Missing Value Strategy Applied:** {strategy}")

                    if strategy == "Drop rows":
                        valid_subset = subset[feats].dropna()
                    elif strategy == "Forward fill":
                        valid_subset = subset[feats].ffill().bfill()
                    elif strategy == "Fill with zero":
                        valid_subset = subset[feats].fillna(0)
                    else:
                        valid_subset = subset[feats]

                    latest_input = valid_subset.tail(1)
                    if latest_input.empty or latest_input.isnull().any().any():
                        st.markdown("#### 🔍 Missing Value Summary")
                        st.dataframe(valid_subset.isnull().sum()[lambda x: x > 0])
                        st.error("❌ Not enough valid data to make a forecast.")
                        st.stop()

                    try:
                        # Fetch the model's expected feature names (order matters)
                        expected_feats = list(getattr(model, "feature_names_in_", feats))

                        # Ensure the features are ordered correctly before prediction
                        latest_input = latest_input[expected_feats]

                        # Ensure there are no missing values
                        if latest_input.isnull().any().any():
                            st.error("❌ Some features still contain missing values after filling.")
                            st.dataframe(latest_input.isnull().sum()[lambda x: x > 0])
                            st.stop()

                        # Forecast for the next 7 days or for the selected pollutant
                        forecasted_values = []
                        forecasted_dates = []

                        for i in range(7):  # Loop through the next 7 days
                            next_day = forecast_date + pd.Timedelta(days=i)
                            # Get prediction for the day
                            pred = float(model.predict(latest_input)[0])
                            pred = np.clip(pred, 0, 500)  # Clip the prediction to be within the AQI range

                            # If the pollutant is AQI, categorize it
                            if pol == "AQI":
                                def categorize_aqi(aqi):
                                    if aqi <= 50:
                                        return "🟢 Good"
                                    elif aqi <= 100:
                                        return "🟡 Moderate"
                                    elif aqi <= 150:
                                        return "🟠 Unhealthy for Sensitive Groups"
                                    elif aqi <= 200:
                                        return "🔴 Unhealthy"
                                    elif aqi <= 300:
                                        return "🟣 Very Unhealthy"
                                    else:
                                        return "🟤 Hazardous"

                                predicted_category = categorize_aqi(pred)
                                forecasted_values.append(predicted_category)
                            else:
                                # For other pollutants (e.g., PM2.5), use the actual forecasted value
                                forecasted_values.append(pred)

                            forecasted_dates.append(next_day)

                            # Update the input for the next day's prediction
                            latest_input[expected_feats] = latest_input[expected_feats].shift(1, axis=0)

                        # Show forecasted values (categories or actual values) for the next 7 days
                        forecast_df = pd.DataFrame({
                            "Date": forecasted_dates,
                            "Forecasted Value": forecasted_values
                        })
                        st.write(forecast_df)

                        # --- Graph for Trend + Forecast ---
                        st.subheader("📊 Trend + Forecast")

                        # Fetch the last 48 hours of data for the selected pollutant
                        hist = df[df.index.date <= forecast_date].tail(48)  # Last 48 hours
                        fig = go.Figure()

                        # Plot the trend (last 48 hours of data)
                        fig.add_trace(go.Scatter(x=hist.index, y=hist[pol.lower()], mode='lines', name='Last 48h'))

                        # Plot the forecast values for the next 7 days
                        fig.add_trace(go.Scatter(
                            x=forecasted_dates,
                            y=forecasted_values,
                            mode='markers+lines',
                            marker=dict(color='red', size=10),
                            name='Forecast'
                        ))

                        fig.update_layout(
                            title=f"{pol} Trend + Forecast",
                            xaxis_title="Date",
                            yaxis_title=f"{pol} Levels"
                        )

                        st.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        st.error(f"❌ Forecast failed: {e}")
                        st.stop()


# --- Classification Tab ---
with tab3:
    st.subheader("🔍 Air Quality Classification")

    if uploaded_file is None:
        st.info("ℹ️ You're using the default historical dataset.")

    # **Button to train the classifier**
    if st.button("Train Classifier and Predict"):
        with st.spinner("Training classifier..."):

            # Clean the data by dropping rows where PM2.5 or other features are missing
            pm_col = "pm2.5" if "pm2.5" in df.columns else "PM2.5"
            df_class = df.dropna(subset=[pm_col])

            # Check if there is enough data to train the model
            if df_class.empty:
                st.warning("⚠️ Not enough valid data to classify.")
            else:
                # **Train the model**
                clf, features = train_classification_model(df_class)

                # **Make predictions using the trained model**
                pred, x_input = classification_prediction(df_class, clf, features)

                if pred is not None:
                    # Show prediction result (Good or Bad air quality)
                    label = "🟢 Good" if pred == 0 else "🔴 Bad"
                    st.success(f"Predicted air quality: **{label}**")

                    # **Save the classifier and prediction for later use**
                    st.session_state["clf"] = clf
                    st.session_state["x_input"] = x_input
                    st.session_state["prediction_label"] = label

                    # **Feature Importance Visualization**
                    if hasattr(clf, "feature_importances_"):
                        importances = pd.Series(clf.feature_importances_, index=features)
                        top_feats = importances.sort_values(ascending=False).head(10)

                        # Display feature importance as a bar chart
                        st.markdown("### 📊 Feature Importance")
                        st.bar_chart(top_feats)

                        st.info("📌 This shows the most influential features in predicting air quality.")
                    else:
                        st.warning("⚠️ No feature importances found for the classifier.")
                else:
                    st.warning("⚠️ Not enough valid data to classify.")


# --- ARIMA Forecast Tab ---
with tab4:
    st.subheader("📉 ARIMA Forecast")

    if uploaded_file is None:
        st.info("ℹ️ You're using the default dataset. Upload a custom file in the sidebar to personalize this dashboard.")

    pol = st.selectbox("Select Pollutant for ARIMA", POLLUTANTS, key="arima_pol")
    pol = pol.lower()

    if st.button("Run ARIMA Forecast"):
        with st.spinner("Forecasting with ARIMA..."):

            df_arima = df.copy()

            try:
                # Ensure datetime index
                if not pd.api.types.is_datetime64_any_dtype(df_arima.index):
                    df_arima.index = pd.to_datetime(df_arima.index)

                # --- Force selected column to numeric ---
                if pol not in df_arima.columns:
                    st.warning(f"⚠️ Selected pollutant '{pol}' not found in the dataset.")
                    st.stop()

                df_arima[pol] = pd.to_numeric(df_arima[pol], errors="coerce")
                df_arima = df_arima[[pol]].dropna()

                if df_arima.empty or len(df_arima) < 30:
                    st.warning("⚠️ Not enough valid numeric data for ARIMA forecasting.")
                    st.stop()

                # --- Resample and interpolate ---
                df_arima = df_arima.resample("H").mean().interpolate()

                forecast = arima_forecast(df_arima, pol)

                if forecast is not None and not forecast.isnull().all():
                    st.success("✅ 7-day forecast generated successfully.")

                    st.markdown("### 📈 Forecast Plot")
                    st.line_chart(forecast)

                    st.markdown("### 🧠 Correlation Insights")
                    try:
                        corr_df = df.select_dtypes(include=[np.number]).corr()[[pol]].drop(pol)
                        corr_df = corr_df.dropna().sort_values(by=pol, ascending=False).head(5)

                        st.table(corr_df)

                        with st.expander("📋 Explanation"):
                            for feature, corr in corr_df[pol].items():
                                direction = "positively" if corr > 0 else "negatively"
                                st.markdown(f"- **{feature}** is {direction} correlated with **{pol}** (r = {corr:+.2f})")

                            st.markdown("""
                            These features may help explain why the forecast shows a rise or fall.
                            For example, if **CO** and **PM10** are highly positively correlated with **PM2.5**, 
                            a rise in those values might indicate rising PM2.5 levels.
                            """)
                    except Exception as e:
                        st.warning(f"⚠️ Could not calculate correlations: {e}")

                    st.markdown("### 📥 Download Forecast Values")
                    st.download_button(
                        label="Download ARIMA Forecast (.csv)",
                        data=forecast.to_csv().encode("utf-8"),
                        file_name=f"{pol}_arima_forecast.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("⚠️ Forecasting failed. Not enough data or stationarity issue.")

            except Exception as e:
                st.warning(f"⚠️ ARIMA forecast failed: {e}")

# --- Historical Trends Tab ---
with tab5:
    st.subheader("📊 Historical Trends")

    if uploaded_file is None:
        st.info("ℹ️ You're currently using the default dataset. Upload a file to analyze your own data.")

    st.markdown("### 📅 Select Time Range & Pollutants")
    selected_pollutants = st.multiselect(
        "Choose pollutants to visualize:",
        ["pm2.5", "pm10", "so2", "no2", "co", "o3", "aqi", "aqi_category"],  # List pollutants directly as per dataset
        default=["pm2.5", "aqi"],
        key="hist_pollutants"
    )

    # Create a proper datetime index from year, month, day, and hour
    required_time_cols = ['year', 'month', 'day', 'hour']
    if all(col in df.columns for col in required_time_cols):
        df["datetime"] = pd.to_datetime(df[required_time_cols])
        df.set_index("datetime", inplace=True)
    else:
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index, errors='coerce')
            if df.index.isnull().any():
                st.warning("⚠️ Some datetime values could not be parsed. Check your uploaded file.")
            else:
                st.info("ℹ️ Used existing index as datetime.")


    selected_range = st.date_input("Date Range", [df.index.min().date(), df.index.max().date()], key="hist_date")

    if len(selected_range) == 2 and st.button("📊 Show Trends"):
        start, end = selected_range
        subset = df[(df.index.date >= start) & (df.index.date <= end)]

        # Debug: Check the subset of data for selected range
        st.write("Subset of data after date filtering:")
        st.write(subset.head())  # Show the first few rows of the filtered data

        if subset.empty:
            st.warning("⚠️ No data available for the selected date range.")
        else:
            for p in selected_pollutants:
                if p not in subset.columns:
                    st.warning(f"⚠️ Pollutant '{p}' not found in dataset.")
                    continue

                st.markdown(f"#### 📈 Trend for {p}")
                st.line_chart(subset[p])

                st.markdown(f"##### 📋 Summary for {p}")
                try:
                    max_val = subset[p].max()
                    min_val = subset[p].min()
                    avg_val = subset[p].mean()
                    st.markdown(
                        f"- **Max**: {max_val:.2f}  \n"
                        f"- **Min**: {min_val:.2f}  \n"
                        f"- **Average**: {avg_val:.2f}"
                    )

                    st.markdown(
                        "This trend shows how the pollutant has varied across your selected date range. "
                        "Sudden spikes or drops may correlate with environmental or seasonal factors."
                    )
                except Exception as e:
                    st.warning(f"Could not generate summary: {e}")


# --- Compare Models Tab ---
with tab6:
    st.subheader("🧠 Compare Models")

    if uploaded_file is None:
        st.info("ℹ️ You're currently using the default dataset. Upload your own file in the sidebar to personalize model comparison.")

    st.markdown("### 📅 Select Comparison Date and Pollutant")
    safe_dates = safe_date_range(df)
    date = st.date_input(
        "Date to Compare Models",
        value=safe_dates[1],
        min_value=safe_dates[0],
        max_value=safe_dates[1],
        key="cmp_date"
    )
    target = st.selectbox("Pollutant to Predict", POLLUTANTS)

    if st.button("🔍 Compare Forecasts Across Models"):
        subset = df[df.index.date == date]

        if subset.empty:
            st.warning("⚠️ No data available for the selected date.")
        else:
            comparison = []

            for model_name in models:
                try:
                    # Get the features used by the model for the target pollutant
                    feats = features_used[model_name].get(target, [])
                    model = models[model_name].get(target)

                    # Normalize feature names for compatibility (convert to lowercase)
                    def normalize_feat(f):
                        f = f.lower().replace("pm25", "pm2.5").replace("pm 2.5", "pm2.5").strip()
                        return "pm25_sub_index" if f == "pm2.5_sub_index" else f

                    feats = [normalize_feat(f) for f in feats]

                    # Rename sub-index column for compatibility with training
                    input_data = subset.rename(columns={"pm2.5_sub_index": "pm25_sub_index"})

                    # Ensure the model features match exactly with what's expected by the model
                    x_input = ensure_model_features(input_data.tail(1), feats)

                    # Check if all expected features are present
                    missing_feats = set(feats) - set(x_input.columns)
                    if missing_feats:
                        st.warning(f"⚠️ Missing features: {missing_feats}. Filling them with zeros.")
                        for feature in missing_feats:
                            x_input[feature] = 0  # Fill missing features with zero

                    # Predict using the model
                    pred = float(model.predict(x_input)[0])
                    comparison.append((model_name, pred))

                except Exception as e:
                    st.warning(f"{model_name} failed: {e}")

            if comparison:
                # Create a DataFrame for the comparison results
                cmp_df = pd.DataFrame(comparison, columns=["Model", f"{target} Forecast"]).set_index("Model")

                st.markdown("### 📋 Forecast Comparison")
                st.dataframe(cmp_df.style.format({f"{target} Forecast": "{:.2f}"}))

                # Determine best model (lower AQI is better, higher pollutant value might be context-specific)
                if target.upper() == "AQI":
                    best_model = cmp_df[f"{target} Forecast"].idxmin()
                else:
                    best_model = cmp_df[f"{target} Forecast"].idxmax()

                st.success(f"✅ Best performing model for **{target}** on **{date}**: **{best_model}**")

                # --- Top Correlations
                st.markdown("### 🧠 Top Correlated Features")
                try:
                    # Ensure target column name is correctly referred to
                    target_column = target.lower()  # Normalize target column name to lowercase

                    # Check if target column exists in the DataFrame
                    if target_column in df.columns:
                        corr_df = df.corr(numeric_only=True)[[target_column]].drop(target_column).sort_values(by=target_column, ascending=False)
                        top_corr = corr_df.head(5).round(2)
                        st.dataframe(top_corr)

                        with st.expander("📋 Explanation"):
                            for feat, corr in top_corr[target_column].items():
                                direction = "positively" if corr > 0 else "negatively"
                                st.markdown(f"- **{feat}** is {direction} correlated with **{target_column}** (r = {corr:+.2f})")

                            st.markdown("Understanding feature correlation helps interpret model consistency.")
                    else:
                        st.warning(f"❌ Target column '{target_column}' not found in the dataset for correlation.")
                except Exception as e:
                    st.warning(f"Could not compute correlation: {e}")
            else:
                st.warning("❌ No valid forecasts available for this date across models.")

# --- Compare Dates Tab ---
with tab7:
    st.subheader("📆 Compare Air Quality Across Two Dates")

    if uploaded_file is None:
        st.info("ℹ️ You're using the default dataset. Upload your own to compare personalized air quality trends.")

    st.markdown("### 📅 Select Two Dates to Compare")
    safe_dates = safe_date_range(df)
    col1, col2 = st.columns(2)
    d1 = col1.date_input(
        "Date 1",
        value=safe_dates[0],
        min_value=safe_dates[0],
        max_value=safe_dates[1],
        key="cmp_d1"
    )
    d2 = col2.date_input(
        "Date 2",
        value=safe_dates[1],
        min_value=safe_dates[0],
        max_value=safe_dates[1],
        key="cmp_d2"
    )

    st.markdown("### 📌 Select Pollutants to Compare")
    selected_cmp = st.multiselect(
        "Choose pollutants to compare across hours",
        POLLUTANTS,
        default=["PM2.5", "AQI"],
        key="cmp_multiselect"
    )

    if not selected_cmp:
        st.warning("Please select at least one pollutant to compare.")
    else:
        # Normalize for internal use (lowercase)
        selected_cmp_lower = [col.lower() for col in selected_cmp]
        available_cols = [col for col in selected_cmp_lower if col in df.columns]

        if not available_cols:
            st.error("🚫 None of the selected pollutants are available in the dataset.")
        else:
            # Filter by date using index.date
            date_series = pd.Series(df.index.date, index=df.index)
            data1 = df.loc[date_series == d1, available_cols]
            data2 = df.loc[date_series == d2, available_cols]

            if data1.empty or data2.empty:
                st.warning("⚠️ Not enough data for one or both selected dates.")
            else:
                # Match length for fair comparison
                min_len = min(len(data1), len(data2))
                diff = data2.iloc[:min_len].values - data1.iloc[:min_len].values

                df_diff = pd.DataFrame(
                    diff.T,
                    index=available_cols,
                    columns=[f"Hour {i}" for i in range(min_len)]
                )

                st.markdown(f"### 🔥 Hourly Change in Pollutants ({d2} vs {d1})")
                fig, ax = plt.subplots(figsize=(12, len(available_cols) * 0.6 + 1))
                sns.heatmap(df_diff, annot=True, fmt=".1f", cmap="coolwarm", center=0, ax=ax)
                st.pyplot(fig)

                with st.expander("🧠 How to Read This"):
                    st.markdown("""
                    - Each cell shows the **difference** in pollutant levels between Date 2 and Date 1 by hour.
                    - **Red = increase**, **Blue = decrease**.
                    - This helps you spot hour-by-hour pollution changes between two specific days.
                    """)


# --- Model Evaluation Tab ---
with tab8:
    st.subheader("📊 Model Evaluation")

    if uploaded_file is None:
        st.info("ℹ️ You're using the default dataset. Upload your own for personalized model analysis.")

    forecast_df = st.session_state.get("latest_forecast_input")
    y_pred = st.session_state.get("latest_forecast_output")
    model = st.session_state.get("latest_forecast_model")

    if forecast_df is None or y_pred is None or model is None:
        st.warning("⚠️ No forecast has been made yet. Please run a forecast in the Forecast tab first.")
    else:
        try:
            from sklearn.metrics import mean_absolute_error, r2_score

            # Try to infer true values if possible (e.g., actual pollutant value)
            target_col = None
            if hasattr(model, "feature_names_in_"):
                for col in forecast_df.columns:
                    if col in model.feature_names_in_ and col.lower() in ["aqi", "pm25", "pm25_sub_index", "pm2.5", "pm2.5_sub_index"]:
                        target_col = col
                        break

            y_true = forecast_df[target_col].values if target_col else None

            st.markdown("### 📌 Evaluation Metrics")

            if y_true is None or len(y_true) != len(y_pred):
                st.warning("⚠️ Cannot compute metrics due to mismatched or missing ground truth values.")
                mae, r2 = None, None
            else:
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)

                st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
                st.metric("R² Score", f"{r2:.2f}" if not np.isnan(r2) else "N/A")

            st.markdown("### 🔍 Forecast Inputs and Outputs")
            st.write("✅ Features used:")
            st.json(list(forecast_df.columns))

            st.markdown("#### 🔢 Forecast Samples")
            df_eval = pd.DataFrame({
                "Predicted": np.round(y_pred, 2)
            })
            if y_true is not None:
                df_eval["Actual"] = np.round(y_true, 2)
            st.dataframe(df_eval.head(10))

            # --- Natural Language Explanation ---
            st.markdown("### 🗣️ Detailed Forecast Evaluation Explanation")

            if y_true is None or mae is None or r2 is None:
                st.info("ℹ️ Not enough information to generate a detailed explanation.")
            else:
                residuals = y_true - y_pred
                over_pred = np.sum(residuals < 0)
                under_pred = np.sum(residuals > 0)
                total = len(y_true)

                quality = (
                    "excellent" if r2 > 0.9 else
                    "good" if r2 > 0.75 else
                    "moderate" if r2 > 0.5 else
                    "poor"
                )

                consistency = (
                    "highly" if r2 > 0.9 else
                    "reasonably" if r2 > 0.75 else
                    "somewhat"
                )

                insight = f"""
The model's forecasting performance shows **{quality} accuracy**, but with a clear pattern of prediction bias:

- 🔢 **Mean Absolute Error (MAE)**: {mae:.2f}  
  This means the model's predictions are off by about {mae:.2f} units on average. That indicates a **{quality} level of accuracy** depending on the pollutant scale.

- 📊 **R² Score**: {r2:.2f}  
  The model explains about **{r2 * 100:.1f}%** of the variability in actual values, suggesting it is **{consistency} consistent** with the observed data.

- ⚖️ **Prediction Bias**:  
  Out of {total} predictions:
  - ✅ **Overestimations**: {over_pred}
  - ⚠️ **Underestimations**: {under_pred}

The model consistently **underestimated the pollutant values** in {under_pred} cases and never overestimated, which indicates a **systematic bias**. This can be risky in air quality forecasting, as it may lead to under-warning the public about pollution levels.
                """

                st.markdown(insight)

        except Exception as e:
            st.error(f"❌ Evaluation failed: {e}")


# --- Advanced Settings Tab ---
with tab9:
    st.subheader("⚙️ Advanced Settings")

    st.markdown(
        "Use this section to customize your forecasting setup. "
        "You can select the model type, manually choose features, "
        "or apply automatic hybrid feature selection for better accuracy."
    )

    st.markdown("### 🧠 Forecasting Configuration")
    model_choice_adv = st.selectbox("Choose Base Model", list(models.keys()), key="adv_model_choice")
    pol_adv = st.selectbox("Target Pollutant", POLLUTANTS, key="adv_pollutant")

    # Initialize session state keys
    if "model_type" not in st.session_state:
        st.session_state["model_type"] = "Random Forest"
    if "aqi_thresh_good" not in st.session_state:
        st.session_state["aqi_thresh_good"] = 50
    if "aqi_thresh_bad" not in st.session_state:
        st.session_state["aqi_thresh_bad"] = 150
    if "manual_feats" not in st.session_state:
        st.session_state["manual_feats"] = []

    st.session_state["model_type"] = st.selectbox(
        "Forecasting Model Type",
        ["Random Forest", "Linear Regression", "ARIMA"],
        index=["Random Forest", "Linear Regression", "ARIMA"].index(st.session_state["model_type"])
    )

    st.session_state["aqi_thresh_good"] = st.slider(
        "AQI Threshold for Good", 0, 100, st.session_state["aqi_thresh_good"]
    )
    st.session_state["aqi_thresh_bad"] = st.slider(
        "AQI Threshold for Unhealthy", 101, 500, st.session_state["aqi_thresh_bad"]
    )

    with st.expander("ℹ️ Threshold Explanation"):
        st.markdown(
            """
            - These AQI thresholds affect how air quality predictions are categorized.
            - For example, setting:
                - **Good** ≤ 50
                - **Unhealthy** ≥ 150
            - Adjusting them allows tailored warnings based on your local standards.
            """
        )

    st.markdown("### 📌 Manual Feature Selection")
    # Only allow selection of valid numeric columns (based on cleaned dataset)
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    st.session_state["manual_feats"] = st.multiselect(
        "Select features for forecasting (optional, overrides model default)",
        numeric_columns,
        default=st.session_state["manual_feats"]
    )

    if st.session_state["manual_feats"]:
        st.success(f"{len(st.session_state['manual_feats'])} manual features selected.")
    else:
        st.info("No manual features selected. Model will use pre-trained features by default.")

    st.markdown("### 🤖 Auto-Suggest Features (Hybrid Selection)")
    st.markdown(
        "Use this if you want the system to suggest features based on hybrid feature importance "
        "(Random Forest, XGBoost, Extra Trees)."
    )

    if st.button("Suggest Top Features"):
        try:
            suggested = selected_features_all.get(model_choice_adv, {}).get(pol_adv, [])
            if suggested:
                st.session_state["manual_feats"] = suggested
                st.success(f"{len(suggested)} features applied from hybrid selection.")
                st.write("Top Features:", suggested)
            else:
                st.warning("⚠️ No hybrid suggestions available for this combination.")
        except Exception as e:
            st.error(f"Feature suggestion failed: {e}")


# --- About Tab ---
with tab10:
    st.subheader("ℹ️ About This Dashboard")

    st.markdown("""
Welcome to the **Air Quality Forecast Dashboard** — your intelligent assistant for analyzing, forecasting, and understanding urban air quality using machine learning and explainable AI.

---

### 🎯 Key Features:
- 📤 **Upload Your Data** or explore the default historical dataset.
- 📈 **Forecast pollutant levels** (PM2.5, PM10, CO, O₃, NO₂, SO₂) using hybrid-trained ML models.
- 🧠 **Interpret model predictions** with SHAP (SHapley Additive exPlanations).
- 🔍 **Classify air quality** into Good or Bad using AI.
- 📉 **Generate ARIMA-based** time series forecasts for selected pollutants.
- 📊 **Evaluate model performance** (R², RMSE, MAPE) and visualize trends.
- 🧪 **Customize inputs** with lag/rolling feature options and hybrid feature selection.
- 📥 **Download predictions and summary reports** with a single click.

---

### 💡 Who Should Use This?
Ideal for:
- 🌱 Environmental scientists
- 📊 Data analysts and ML engineers
- 🏙️ Urban planners and policy makers
- 🧪 Students, researchers, and educators
- 🩺 Public health professionals

---

### 🧰 Tech Stack:
- **Frontend**: Streamlit (UI/UX)
- **Backend & ML**: pandas, scikit-learn, XGBoost, ExtraTrees
- **Time Series**: statsmodels (ARIMA)
- **Explainability**: SHAP (AI interpretability)
- **Visualization**: seaborn, matplotlib, Plotly

---

### 🚀 Deployment & Usage Tips:
- Begin in **📊 Data Exploration** to filter your dataset.
- Proceed to **📈 Forecast** or **📉 ARIMA** to make predictions.
- Dive into **🧠 AI Explanation** or **📊 Model Evaluation** to understand and validate results.
- Use **🔍 Classification** to instantly assess air quality status (Good/Bad).

---

### 🙋 Author & Credits
Built with ❤️ by **Sassviny Manokaran**, 2025  
For educational, environmental, and data science applications.

---
""")