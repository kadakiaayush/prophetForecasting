import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from prophet import Prophet
from datetime import datetime
from statsmodels.tsa.stattools import adfuller

# Load and preprocess data
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start="2000-01-01", end=datetime.today().strftime('%Y-%m-%d'))
    data.reset_index(inplace=True)
    return data

def preprocess_data(data):
    data = data[['Date', 'Adj Close']].rename(columns={'Date': 'ds', 'Adj Close': 'y'})
    # Convert to numeric and handle errors, then drop NaNs
    data['y'] = pd.to_numeric(data['y'], errors='coerce')
    data.dropna(subset=['y'], inplace=True)
    return data

# Fit Prophet model
def fit_prophet_model(data):
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.fit(data)
    return model

# Make forecasts
def make_forecasts(model, periods):
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast

# Plot interactive forecast with Plotly
def plot_interactive_forecast(data, forecast):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.15,
                        subplot_titles=("Stock Prices with Technical Indicators", "RSI"))

    # Plot actual and forecasted data
    fig.add_trace(go.Scatter(x=data['ds'], y=data['y'], mode='lines', name='Actual', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecasted', line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill=None, mode='lines', line=dict(color='pink'), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', mode='lines', line=dict(color='pink'), name='Confidence Interval'), row=1, col=1)

    # Bollinger Bands
    rolling_mean = data['y'].rolling(window=20).mean()
    rolling_std = data['y'].rolling(window=20).std()
    upper_band = rolling_mean + (rolling_std * 2)
    lower_band = rolling_mean - (rolling_std * 2)
    fig.add_trace(go.Scatter(x=data['ds'], y=upper_band, mode='lines', name='Upper Bollinger Band', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data['ds'], y=lower_band, mode='lines', name='Lower Bollinger Band', line=dict(color='green')), row=1, col=1)

    # RSI Calculation
    delta = data['y'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    fig.add_trace(go.Scatter(x=data['ds'], y=rsi, mode='lines', name='RSI', line=dict(color='purple')), row=2, col=1)

    fig.update_layout(height=600, title='Stock Price Forecast with Technical Indicators', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)

# Calculate and display technical indicators and metrics
def calculate_metrics(data):
    # Sharpe Ratio
    daily_returns = data['y'].pct_change()
    mean_return = daily_returns.mean()
    std_dev = daily_returns.std()
    sharpe_ratio = (mean_return / std_dev) * np.sqrt(252)

    # Volatility
    volatility = std_dev * np.sqrt(252)

    # Augmented Dickey-Fuller test for stationarity
    adf_test = adfuller(data['y'].dropna())
    adf_statistic = adf_test[0]
    adf_p_value = adf_test[1]

    st.subheader('Technical and Statistical Metrics')
    st.write(f"**Sharpe Ratio**: {sharpe_ratio:.2f} ")
    st.write(f"**Volatility (Annualized)**: {volatility:.2f} ")
    st.write(f"**ADF Statistic**: {adf_statistic:.2f} ")
    st.write(f"**ADF p-value**: {adf_p_value:.2f} ")

# Description tab
def show_description():
    st.markdown("""
    ## Project Description
    This project offers a comprehensive analysis of the S&P 500, NASDAQ, & Dow Jones indices, featuring time series forecasting using the Prophet model and various financial metrics. The analysis includes:

    - **Trend Analysis**: Understanding long-term price movements.
    - **Seasonality Effects**: Identifying periodic fluctuations.
    - **Technical Indicators**: Such as Bollinger Bands and RSI for market momentum and volatility assessment.

    ### Key Components and Mathematical Model
    The core model, Prophet, decomposes the time series into:
    - **Trend**: \( g(t) \) for long-term changes.
    - **Seasonality**: \( s(t) \) for periodic fluctuations.
    - **Error**: \( \epsilon_t \) for residuals.

    The equation used is:
    \[
    y(t) = g(t) + s(t) + \epsilon_t
    \]

    ### Financial Analysis
    - **Sharpe Ratio**: A measure of risk-adjusted return. A higher value indicates better risk-adjusted performance.
    - **Volatility**: Indicates the degree of price variation and associated risk. A measure of price variability over time. Higher volatility indicates higher risk.
    - **ADF Test**: Used to test if a time series is stationary. A more negative value indicates stronger rejection of non-stationarity.
    - **ADF p-value**: Probability that the time series is non-stationary. A lower value indicates stronger evidence against non-stationarity.

    This project aims to provide investors and analysts with an advanced toolset for making data-driven decisions based on financial and statistical analyses.
    """)

# Streamlit app
def main():
    st.set_page_config(page_title="Advanced S&P 500 Forecasting", layout="wide")

    st.title('Advanced Stock Market Forecasting with Prophet')
    menu = ["Forecasting", "Project Description"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Forecasting":
        st.subheader("Stock Forecasting")

        # Input for ticker symbol
        ticker = st.text_input('Enter ticker symbol (e.g. ^GSPC for S&P 500, ^IXIC for NASDAQ, ^DJI for Dow Jones):', value='^GSPC')

        # Load and display data
        data_load_state = st.text('Loading data...')
        data = load_data(ticker)
        data_load_state.text('Loading data... done!')

        # Display raw data in a more readable format
        st.subheader('Raw Data Overview')
        st.dataframe(data.tail(10).style.format({'Date': '{:%Y-%m-%d}', 'Adj Close': '${:.2f}'}))

        # Preprocess data
        data = preprocess_data(data)

        # Fit Prophet model
        model = fit_prophet_model(data)

        # Forecasting
        forecast_horizon = st.slider('Forecast horizon (days):', min_value=30, max_value=365*5, value=365*2)
        forecast = make_forecasts(model, periods=forecast_horizon)

        # Display forecasts
        st.subheader('Forecasts')
        st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

        # Plot interactive forecast
        st.subheader('Interactive Forecast Plot with Technical Indicators')
        plot_interactive_forecast(data, forecast)

        # Display metrics and technical indicators
        calculate_metrics(data)

    elif choice == "Project Description":
        show_description()

if __name__ == '__main__':
    main()
