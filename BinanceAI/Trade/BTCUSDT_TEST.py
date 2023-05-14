import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import requests
import json
import datetime
import time
import telegram
import asyncio

# Create and fit the scaler on the data

while True:
                
                bot = telegram.Bot(token='########')

                def show_current_time():
                        now = datetime.datetime.now()
                        print("current time: ", now.strftime("%H:%M:%S"))

                async def my_function():
                    # Loading the best trained model for prediction
                    best_model = keras.models.load_model("best_model.h5")

                    # Getting data for the last 24 hours via the Binance API
                    response = requests.get('https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=5m&limit=1440')
                    raw_data = json.loads(response.text)

                    # Converting Data to a DataFrame
                    df = pd.DataFrame(raw_data, columns=['Datetime', 'Open', 'High', 'Low', 'Price', 'Volume', 'CloseTime', 'QuoteAssetVolume', 'NumberOfTrades', 'TakerBuyBaseAssetVolume', 'TakerBuyQuoteAssetVolume', 'Ignore'])
                    df = df[['Datetime', 'Price', 'Volume']]
                    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='ms')
                    df = df.set_index('Datetime')

                    scaler = MinMaxScaler()
                    scaler.fit(df[['Price', 'Volume']])

                    # Getting the current price and time
                    current_price = df['Price'].iloc[-1]

                    # Data normalization
                    df[['Price', 'Volume']] = scaler.transform(df[['Price', 'Volume']])

                    # Formation of a sequence 
                    sequence = df.tail(1000).values.reshape(1, 1, 2)

                    # Price prediction for the next 5 minutes
                    prediction = best_model.predict(sequence)

                    # Deciphering the normalized forecast into the original price scale
                    prediction = scaler.inverse_transform(np.concatenate((prediction, np.zeros((1, 1))), axis=1))[:, 0]

                    print('Price forecast for the next 5 minutes:', prediction,'Current price:', current_price,)
                    show_current_time()
                    
                    # Start timer for 5 minutes
                    def timer(minutes):
                        seconds = minutes * 60
                        while seconds > 0:
                            minutes, sec = divmod(seconds, 60)
                            print(f"Time left: {minutes:02d}:{sec:02d}")
                            time.sleep(1)
                            seconds -= 1
                        print("Time is over!")

                    timer(5) 

                    
                    # Getting data for the last 24 hours via the Binance API
                    response = requests.get('https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=5m&limit=1440')
                    raw_data = json.loads(response.text)


                    # Converting Data to a DataFrame
                    df = pd.DataFrame(raw_data, columns=['Datetime', 'Open', 'High', 'Low', 'Price', 'Volume', 'CloseTime', 'QuoteAssetVolume', 'NumberOfTrades', 'TakerBuyBaseAssetVolume', 'TakerBuyQuoteAssetVolume', 'Ignore'])
                    df = df[['Datetime', 'Price', 'Volume']]
                    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='ms')
                    df = df.set_index('Datetime')

                    scaler = MinMaxScaler()
                    scaler.fit(df[['Price', 'Volume']])

                    # Getting the current price and time
                    current_price = df['Price'].iloc[-1]

                    def check_prediction(current_price, predicted_price):
                        if current_price < predicted_price - 10:
                            print('The predicted price is more than $1 higher than the current price. -1 point')
                            return -1
                        elif current_price > predicted_price + 10:
                            print('The predicted price is more than $1 below the current price. -1 point')
                            return -1
                        else:
                            print('The predicted price is within the $1 margin of error. +1 point')
                            return 1

                        
                    score = check_prediction(float(current_price), prediction)
                    print('Point: ', score) 

                    # Send a message to a channel
                    message = f"BTCUSDT Price Prediction Next 5 Minutes: {prediction}. Current price: {current_price}.           {score}"
                    await bot.send_message(chat_id='####', text=message)

                async def main():
                    show_current_time()
                    await my_function()

                if __name__ == '__main__':
                    asyncio.run(main())
                    print('Очки: ') 