import numpy as np
import pandas as pd
from binance.client import Client
import os






# Function for getting historical data and saving it to a table
def get_and_save_data(symbol, interval, start_date, end_date, save_file):
    # Binance API Options
    api_key = ''
    api_secret = ''
    client = Client(api_key, api_secret)

    # Getting historical data on the price and trading volume for a symbol pair
    klines = client.get_historical_klines(symbol, interval, start_date, end_date)

    # We form an array with data, where each line contains the price, trading volume and date at a certain point in time
    data = []
    for kline in klines:
        data.append([pd.Timestamp(kline[0], unit='ms'), float(kline[1]), float(kline[5])])
    data = np.array(data)

    # We save the data in the table, if it does not exist, then create it
    if not os.path.isfile(save_file):
        df = pd.DataFrame(data, columns=['Datetime', 'Price', 'Volume'])
        df.to_csv(save_file, index=False)
    else:
        df = pd.read_csv(save_file)
        df = df.append(pd.DataFrame(data, columns=['Datetime', 'Price', 'Volume']))
        df.to_csv(save_file, index=False)

    return df


# Options for getting historical data
symbol = 'BTCUSDT'
interval = Client.KLINE_INTERVAL_5MINUTE
start_date = '1 Jan, 2017'
end_date = 'now'
save_file = 'BTCUSDT.csv'

# Getting historical data and saving it to a table
data = get_and_save_data(symbol, interval, start_date, end_date, save_file)


