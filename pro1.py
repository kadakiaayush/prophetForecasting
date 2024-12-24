import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
import plotly.graph_objs as go
from fredapi import Fred
from plotly.subplots import make_subplots

# Set up FRED API key (replace 'your_fred_api_key' with your actual key)
FRED_API_KEY = 'your_fred_api_key'
fred = Fred(api_key=FRED_API_KEY)

# Load and preprocess data
@st.cache_data
def load_yf_data(ticker):
    try:
        data = yf.download(ticker, start="2000-01-01", end=datetime.today().strftime('%Y-%m-%d'))
        if data.empty:
            st.error("No data found for the ticker. Please try another symbol.")
            return pd.DataFrame()
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error fetching data from Yahoo Finance: {e}")
        return pd.DataFrame()

@st.cache_data
def load_fred_data(series_id):
    try:
        data = fred.get_series(series_id).reset_index()
        data.columns = ['Date', 'Value']
        data['Date'] = pd.to_datetime(data['Date'])
        return data
    except Exception as e:
        st.error(f"Error fetching data from FRED: {e}")
        return pd.DataFrame()

def preprocess_data(data, column_name='Adj Close'):
    if column_name not in data.columns:
        st.error(f"Column '{column_name}' not found in data.")
        return pd.DataFrame()
    
    data = data[['Date', column_name]].rename(columns={'Date': 'ds', column_name: 'y'})
    data['y'] = pd.to_numeric(data['y'], errors='coerce')
    data.dropna(inplace=True)
    return data

# Fit Prophet model
def fit_prophet_model(data):
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.fit(data)
    return model

# Forecasting
def make_forecasts(model, periods):
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast

# Sharpe Ratio and Volatility
def calculate_sharpe_and_volatility(data):
    daily_returns = data['y'].pct_change().dropna()
    mean_return = daily_returns.mean()
    std_dev = daily_returns.std()
    sharpe_ratio = (mean_return / std_dev) * np.sqrt(252)
    volatility = std_dev * np.sqrt(252)
    return sharpe_ratio, volatility

# ADF Test
def adf_test(data):
    adf_result = adfuller(data['y'].dropna())
    return adf_result[0], adf_result[1]

# Plot Interactive Forecast
def plot_interactive_forecast(data, forecast):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("Actual vs Forecasted", "RSI and Bollinger Bands"))

    # Actual vs Forecasted
    fig.add_trace(go.Scatter(x=data['ds'], y=data['y'], mode='lines', name='Actual'), row=1, col=1)
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'), row=1, col=1)
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill=None, mode='lines', name='Lower Bound', line=dict(color='pink')), row=1, col=1)
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', mode='lines', name='Upper Bound', line=dict(color='pink')), row=1, col=1)

    # Bollinger Bands and RSI
    rolling_mean = data['y'].rolling(window=20).mean()
    rolling_std = data['y'].rolling(window=20).std()
    upper_band = rolling_mean + (rolling_std * 2)
    lower_band = rolling_mean - (rolling_std * 2)
    fig.add_trace(go.Scatter(x=data['ds'], y=upper_band, mode='lines', name='Upper Band', line=dict(color='orange')), row=2, col=1)
    fig.add_trace(go.Scatter(x=data['ds'], y=lower_band, mode='lines', name='Lower Band', line=dict(color='green')), row=2, col=1)

    delta = data['y'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    fig.add_trace(go.Scatter(x=data['ds'], y=rsi, mode='lines', name='RSI', line=dict(color='purple')), row=2, col=1)

    fig.update_layout(height=800, title="Forecast with Technical Indicators", xaxis_title="Date", yaxis_title="Value")
    st.plotly_chart(fig)

# Streamlit App
def main():
    st.title("Market Volatility Forecasting with Prophet")
    st.sidebar.header("Options")

    menu = ["Forecasting", "FRED Data Analysis", "Description"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Forecasting":
        st.subheader("Yahoo Finance Forecasting")
        ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT):", value="AAPL")

        data_load_state = st.text("Loading data...")
        data = load_yf_data(ticker)
        data_load_state.text("Loading data... done!")

        if not data.empty:
            st.write("Raw Data")
            st.dataframe(data.tail())

            preprocessed_data = preprocess_data(data)
            if not preprocessed_data.empty:
                model = fit_prophet_model(preprocessed_data)
                forecast_horizon = st.slider("Forecast Horizon (days):", min_value=30, max_value=365, value=180)
                forecast = make_forecasts(model, forecast_horizon)

                st.write("Forecast Data")
                st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

                sharpe_ratio, volatility = calculate_sharpe_and_volatility(preprocessed_data)
                adf_stat, adf_p = adf_test(preprocessed_data)

                st.write(f"**Sharpe Ratio**: {sharpe_ratio:.2f}")
                st.write(f"**Volatility**: {volatility:.2f}")
                st.write(f"**ADF Statistic**: {adf_stat:.2f}")
                st.write(f"**ADF p-value**: {adf_p:.2f}")

                plot_interactive_forecast(preprocessed_data, forecast)

    elif choice == "FRED Data Analysis":
        st.subheader("FRED Data Analysis")
        fred_series_id = st.text_input("Enter FRED Series ID (e.g., DGS10 for 10-Year Treasury):", value="DGS10")

        data_load_state = st.text("Loading FRED data...")
        fred_data = load_fred_data(fred_series_id)
        data_load_state.text("Loading FRED data... done!")

        if not fred_data.empty:
            st.write("FRED Data")
            st.dataframe(fred_data.tail())

            preprocessed_data = preprocess_data(fred_data, column_name="Value")
            if not preprocessed_data.empty:
                model = fit_prophet_model(preprocessed_data)
                forecast_horizon = st.slider("Forecast Horizon (days):", min_value=30, max_value=365, value=180)
                forecast = make_forecasts(model, forecast_horizon)

                st.write("Forecast Data")
                st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

                plot_interactive_forecast(preprocessed_data, forecast)

    elif choice == "Description":
        st.subheader("Project Description")
        st.write("""
            This app leverages Meta's Prophet for time-series modeling to forecast market volatility, 
            analyze stock prices, and evaluate economic indicators from the FRED API.
        """)

if __name__ == "__main__":
    main()
