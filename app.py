import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

st.title("📈 LSTM-basert aksjeprediksjon")

# --- Brukervalg
ticker = st.text_input("Velg aksje (eks: AAPL)", value="AAPL")
start = st.date_input("Startdato", pd.to_datetime("2018-01-01"))
end = st.date_input("Sluttdato", pd.to_datetime("2024-12-31"))

if st.button("Hent og tren modell"):

    # --- Hent data
    data = yf.download(ticker, start=start, end=end)
    close_data = data['Close'].values.reshape(-1, 1)

    # --- Skaler data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(close_data)

    # --- Lag treningsdata
    look_back = 60
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # --- LSTM-modell
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    # --- Forutsi neste verdi
    last_60_days = scaled_data[-look_back:]
    X_pred = last_60_days.reshape(1, look_back, 1)
    predicted_scaled = model.predict(X_pred)
    predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]

    st.success(f"📊 Forventet neste dags pris: ${predicted_price:.2f}")

    # --- Plot resultater
    fig, ax = plt.subplots()
    ax.plot(data.index, data['Close'], label='Historisk pris')
    ax.axhline(predicted_price, color='orange', linestyle='--', label='Predikert pris')
    ax.set_title(f"{ticker} – Historikk og prediksjon")
    ax.legend()
    st.pyplot(fig)
