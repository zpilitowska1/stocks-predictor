import streamlit as st
import pandas as pd
import altair as alt
import joblib
import numpy as np
from datetime import timedelta

# PAGE CONFIG
st.set_page_config(
    page_title="SmartPredict - Stock Classifier",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 SmartPredict - Stock Direction Classifier")


# --- Load Dataset ---
@st.cache_data
def load_data():
    # Change path to your actual CSV file
    df = pd.read_csv(
        r"stock_metrics_downloaded.csv"
    )
    # Make sure date column is datetime and set as index
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df


df = load_data()


# --- Company & model info (update paths to your actual models and scalers) ---
def get_company_dict():
    return {
        "AAPL": {
            "clf_model": r"models\Classification\AAPL_XGB.pkl",
            "reg_model": r"models\Regression\AAPL_LR.pkl",
            "scaler": r"Scalers\AAPL_SCALER_CLF.save",
            "accuracy": 55,
        },
        "GOOG": {
            "clf_model": r"models\Classification\GOOG_XGB.pkl",
            "reg_model": r"models\Regression\GOOG_LR.pkl",
            "scaler": r"Scalers\GOOG_SCALER_CLF.save",
            "accuracy": 53,
        },
        "QCOM": {
            "clf_model": r"models\Classification\QCOM_XGB.pkl",
            "reg_model": r"models\Regression\QCOM_LR.pkl",
            "scaler": r"Scalers\QCOM_SCALER_CLF.save",
            "accuracy": 52,
        },
    }


# Sidebar controls
st.sidebar.header("Select Options")
company_dict = get_company_dict()
selected_company = st.sidebar.selectbox("Select Company", list(company_dict.keys()))

# Filter data by selected company (ticker)
df_company = df[df["ticker"] == selected_company].copy()
if df_company.empty:
    st.error(f"No data found for {selected_company}.")
    st.stop()

# Extract year list
df_company["year"] = df_company.index.year
years = sorted(df_company["year"].unique(), reverse=True)
selected_year = st.sidebar.selectbox("Select Year", years)

df_filtered = df_company[df_company["year"] == selected_year].copy()
if df_filtered.empty:
    st.warning("No data for selected year.")
    st.stop()

# Date selection for forecasting
min_date = df_filtered.index.min().date()
max_date = df_filtered.index.max().date()
selected_date = st.sidebar.date_input(
    "Select Forecast Date",
    value=max_date,
    min_value=min_date,
    max_value=max_date,
)

# Line color for charts
line_color = st.sidebar.selectbox(
    "Select Line Color", ["blue", "green", "red", "orange", "purple", "black"]
)

# Features used for modeling
features = [
    "close_lag",
    "close_mean_10_days",
    "close_std_10_days",
    "close_max_10_days",
    "close_mean_30_days",
    "close_std_30_days",
    "close_max_30_days",
    "ema_close_10_days",
    "ema_close_30_days",
    "bb_upper_20",
    "bb_lower_20",
    "bb_middle_20",
    "bb_upper_50",
    "bb_lower_50",
    "bb_middle_50",
    "eps_estimate",
    "eps_actual",
    "eps_surprise",
    "surprise_percent",
]

# Load models and scaler
model_info = company_dict[selected_company]

try:
    clf_model = joblib.load(model_info["clf_model"])
    reg_model = joblib.load(model_info["reg_model"])
    scaler = joblib.load(model_info["scaler"])
except Exception as e:
    st.error(f"Error loading models/scaler: {e}")
    st.stop()

# Prepare data for prediction: drop rows with NaNs in features
df_filtered = df_filtered.dropna(subset=features)

# Scale features
X = df_filtered[features].values
X_scaled = scaler.transform(X)

# Predict classification direction and regression close price
df_filtered["Prediction"] = clf_model.predict(X_scaled)
df_filtered["Direction"] = df_filtered["Prediction"].map({1: "📈 Up", 0: "📉 Down"})

if hasattr(clf_model, "predict_proba"):
    proba = clf_model.predict_proba(X_scaled)
    df_filtered["Down Probability"] = proba[:, 0]
    df_filtered["Up Probability"] = proba[:, 1]
else:
    df_filtered["Down Probability"] = np.nan
    df_filtered["Up Probability"] = np.nan

df_filtered["Predicted_Close"] = reg_model.predict(X_scaled)

# Show company info
st.markdown(
    f"### Company: {selected_company} | Year: {selected_year} | Model Accuracy: {model_info['accuracy']}%"
)

# Show recent actual vs predicted close price chart
st.subheader("Actual Target vs Predicted Close Prices")
price_chart_df = (
    df_filtered[["target", "Predicted_Close"]]
    .reset_index()
    .melt(id_vars="date", var_name="Type", value_name="Price")
)

chart = (
    alt.Chart(price_chart_df)
    .mark_line()
    .encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("Price:Q", title="Price"),
        color="Type:N",
        tooltip=[alt.Tooltip("date:T", title="Date"), "Price"],
    )
    .properties(height=400)
    .interactive()
)

st.altair_chart(chart, use_container_width=True)


# Show classification predictions table with probabilities for selected date and after
st.subheader("Classification Predictions with Probabilities")
forecast_start_date = pd.to_datetime(selected_date)

# Get the last available feature row before or on the selected date
hist_data = df_filtered[df_filtered.index <= forecast_start_date]
if hist_data.empty:
    st.warning("No historical data before the selected date.")
else:
    last_features = hist_data.iloc[-1][features].to_frame().T
    # Repeat the last feature vector for next 5 trading days
    future_features = pd.concat([last_features] * 5, ignore_index=True)
    future_scaled = scaler.transform(future_features)
    future_preds = clf_model.predict(future_scaled)
    future_proba = (
        clf_model.predict_proba(future_scaled)
        if hasattr(clf_model, "predict_proba")
        else np.full((5, 2), np.nan)
    )
    future_prices = reg_model.predict(future_scaled)

    future_dates = [forecast_start_date + timedelta(days=i + 1) for i in range(5)]

    forecast_df = pd.DataFrame(
        {
            "Date": future_dates,
            "Direction": ["📈 Up" if p == 1 else "📉 Down" for p in future_preds],
            "Down Probability": future_proba[:, 0],
            "Up Probability": future_proba[:, 1],
            "Predicted Close": future_prices,
        }
    )

    # Format and display
    st.dataframe(
        forecast_df.style.format(
            {
                "Down Probability": "{:.1%}",
                "Up Probability": "{:.1%}",
                "Predicted Close": "${:,.2f}",
            }
        ),
        use_container_width=True,
    )
