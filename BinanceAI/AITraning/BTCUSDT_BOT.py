import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import requests
import json


# Loading and preparing data
df = pd.read_csv('BTCUSDT.csv', parse_dates=['Datetime'])
df = df.set_index('Datetime')
scaler = MinMaxScaler()
df[['Price', 'Volume']] = scaler.fit_transform(df[['Price', 'Volume']])
X = []
y = []
for i in range(1, len(df)):
    X.append(df.iloc[i-1:i, :].values)
    y.append(df.iloc[i, 0])
X = np.array(X)
y = np.array(y)

# Model creation
model = keras.Sequential([
    keras.layers.LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(units=50, return_sequences=True),
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(units=50),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(units=1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Model Training and Saving the Best Trained Epochs
checkpoint_cb = keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True)
history = model.fit(X, y, epochs=100, batch_size=32, validation_split=0.1, callbacks=[checkpoint_cb])

# Loading the best trained model for prediction
best_model = keras.models.load_model("best_model.h5")

# Getting data via the Binance API
response = requests.get('https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=5m&limit=1440')
raw_data = json.loads(response.text)

# Converting Data to a DataFrame
df = pd.DataFrame(raw_data, columns=['Datetime', 'Open', 'High', 'Low', 'Price', 'Volume', 'CloseTime', 'QuoteAssetVolume', 'NumberOfTrades', 'TakerBuyBaseAssetVolume', 'TakerBuyQuoteAssetVolume', 'Ignore'])
df = df[['Datetime', 'Price', 'Volume']]
df['Datetime'] = pd.to_datetime(df['Datetime'], unit='ms')
df = df.set_index('Datetime')

# Data normalization
df[['Price', 'Volume']] = scaler.transform(df[['Price', 'Volume']])

# Formation of a sequence 
sequence = df.tail(1).values.reshape(1, 1, 2)

# Price prediction for the next 5 minutes
prediction = best_model.predict(sequence)

# Deciphering the normalized forecast into the original price scale
prediction = scaler.inverse_transform(np.concatenate((prediction, np.zeros((1, 1))), axis=1))[:, 0]

print('Price forecast for the next 5 minutes:', prediction)
