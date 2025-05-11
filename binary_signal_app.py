
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands
from sklearn.ensemble import RandomForestClassifier
import requests
import datetime

# ------------------- Telegram Config -------------------
TELEGRAM_TOKEN = '7557174507:AAFSmFW5nxJ-fLOPS-B_wi0uT5wkQ5-PEx8'
CHAT_ID = '1278635048'

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print("Telegram Error:", e)

# ------------------- Feature Engineering -------------------
def add_indicators(df):
    df['ema_9'] = EMAIndicator(df['Close'], window=9).ema_indicator()
    df['rsi'] = RSIIndicator(df['Close']).rsi()
    df['macd'] = MACD(df['Close']).macd_diff()
    bb = BollingerBands(df['Close'])
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    return df.dropna()

# ------------------- Model Training -------------------
def train_model(df):
    X = df[['ema_9', 'rsi', 'macd', 'bb_high', 'bb_low']]
    y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    return model

# ------------------- Signal Prediction -------------------
def predict_signal(df, model):
    X = df[['ema_9', 'rsi', 'macd', 'bb_high', 'bb_low']].tail(1)
    prediction = model.predict(X)[0]
    return "CALL 📈" if prediction == 1 else "PUT 📉"

# ------------------- Load Data -------------------
def load_data(symbol):
    df = yf.download(symbol, interval='1m', period='2d')
    df.reset_index(inplace=True)
    return df

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="Binary Signal Bot", layout="centered", page_icon="📉")
st.title("📊 Binary Trading Signal Bot")
selected_symbol = st.selectbox("Select Forex Pair", ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "AUDUSD=X", "USDCAD=X"])

with st.spinner("Fetching data and generating signal..."):
    df = load_data(selected_symbol)
    df = add_indicators(df)
    model = train_model(df)
    signal = predict_signal(df, model)

    st.subheader(f"🔔 {selected_symbol.replace('=X', '')} Signal: {signal}")
    send_telegram_message(f"{selected_symbol.replace('=X', '')} Signal: {signal}")

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['Datetime'], open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name='Candles'))
    fig.update_layout(title=f"{selected_symbol.replace('=X', '')} Chart with Signal", xaxis_title="Time", yaxis_title="Price")
    st.plotly_chart(fig)

    st.download_button("Download Recent Data", df.to_csv(index=False), file_name="recent_data.csv")

st.caption("Next candle updates every minute. Powered by Streamlit & scikit-learn.")
