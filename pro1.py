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
    try:
        data = yf.download(ticker, start="2000-01-01", end=datetime.today().strftime('%Y-%m-%d'))
        if data.empty:
            st.error(f"No data found for ticker symbol '{ticker}'. Please try a valid symbol.")
            return pd.DataFrame()

        # Use 'Adj Close' if available, otherwise fall back to 'Close'
        if 'Adj Close' in data.columns:
            data['Close'] = data['Adj Close']
        elif 'Close' not in data.columns:
            st.error("No suitable price column found (e.g., 'Adj Close' or 'Close').")
            return pd.DataFrame()

        data.reset_index(inplace=True)
        return data

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def preprocess_data(data):
    if data.empty:
        st.error("Data is empty. Cannot preprocess.")
        return pd.DataFrame()

    try:
        # Ensure 'Date' and 'Close' columns exist
        if 'Date' not in data.columns or 'Close' not in data.columns:
            st.error("Required columns 'Date' or 'Close' are missing.")
            return pd.DataFrame()

        # Convert 'Date' to datetime
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

        # Filter necessary columns and rename them
        data = data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

        # Ensure 'y' is numeric and drop NaN values
        data['y'] = pd.to_numeric(data['y'], errors='coerce')
        data.dropna(subset=['y'], inplace=True)

        if data.empty:
            st.error("No valid data available after preprocessing. Please check the data source.")
            return pd.DataFrame()

        return data

    except Exception as e:
        st.error(f"Error during preprocessing: {str(e)}")
        return pd.DataFrame()

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
    fig.add_trace(go.Scatter(x=data['ds'], y=upper_band, mode='lines', name='Upper Bollinger Band', line=dict(color='orange')), row=2, col=1)
    fig.add_trace(go.Scatter(x=data['ds'], y=lower_band, mode='lines', name='Lower Bollinger Band', line=dict(color='green')), row=2, col=1)

    # RSI Calculation
    delta = data['y'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    fig.add_trace(go.Scatter(x=data['ds'], y=rsi, mode='lines', name='RSI', line=dict(color='purple')), row=2, col=1)

    fig.update_layout(height=600, title='Stock Price Forecast with Technical Indicators', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)

# Streamlit app
def main():
    st.set_page_config(page_title="Advanced Stock Forecasting", layout="wide")
    st.title('Advanced Stock Market Forecasting with Prophet')
    menu = ["Forecasting", "Project Description"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Forecasting":
        st.subheader("Stock Forecasting")
        ticker = st.text_input('Enter ticker symbol (e.g., ^GSPC for S&P 500):', value='^GSPC')

        # Load and display data
        data_load_state = st.text('Loading data...')
        data = load_data(ticker)
        data_load_state.text('Loading data... done!')

        if not data.empty:
            st.subheader('Raw Data Overview')
            st.dataframe(data.tail(10))

            # Preprocess data
            preprocessed_data = preprocess_data(data)

            if not preprocessed_data.empty:
                model = fit_prophet_model(preprocessed_data)

                # Forecasting
                forecast_horizon = st.slider('Forecast horizon (days):', min_value=30, max_value=365 * 5, value=365 * 2)
                forecast = make_forecasts(model, periods=forecast_horizon)

                # Display forecasts
                st.subheader('Forecasts')
                st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

                # Plot interactive forecast
                st.subheader('Interactive Forecast Plot with Technical Indicators')
                plot_interactive_forecast(preprocessed_data, forecast)
            else:
                st.error("Processed data is empty. Please check the data source or preprocessing steps.")

if __name__ == '__main__':
    main()
