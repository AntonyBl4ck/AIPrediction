# AIPrediction
Welcome

Let's start with BTCUSDT_DATA.py. This is the code that parses the price data of the cryptocurrency pair that we specify, as well as the time period and at the end are saved in the cvs format. Due to the fact that in my code the period is from 2017 to this day, the data is parsed for 5 minutes. Don't forget to enter "api key" and "api secret"

Once we have the historical data running, we can move on to BTCUSDT_BOT.py.
The model used for price prediction in this code is a Long Short-Term Memory (LSTM) neural network. LSTM is a type of recurrent neural network (RNN) that is well-suited for sequential data analysis, such as time series data.
The algorithm used in this code can be described as follows:
1.Loading and preparing data: Historical Bitcoin price and volume data is loaded from a CSV file. The data is then indexed by the datetime column and scaled using the MinMaxScaler from sklearn. The scaled data is split into input (X) and output (y) arrays. X is created by selecting the previous row as input and y is created by selecting the current price as the target.
2.Model creation: The model is created using the Keras Sequential API. It consists of three LSTM layers with 50 units each, followed by dropout layers to prevent overfitting, and a dense layer with 1 unit for the output. The input shape of the model is determined by the shape of the input data (X).
3.Model training: The model is trained using the compiled model object. The loss function used is mean squared error (MSE), and the optimizer used is Adam. The training is performed for 100 epochs with a batch size of 32. The training data is split into a training set and a validation set (10% of the data) for monitoring the model's performance during training. The best trained model based on validation loss is saved using a ModelCheckpoint callback.
4.Loading the best trained model: The best trained model (saved as "best_model.h5") is loaded for making predictions.
5.Data retrieval: Data for making predictions is obtained from the Binance API. The API returns raw data in JSON format, which is converted to a DataFrame with relevant columns ('Datetime', 'Price', 'Volume'). The 'Datetime' column is converted to datetime format and set as the index.
6.Data normalization: The newly obtained data is normalized using the same MinMaxScaler that was used during training.
7.Sequence formation: A sequence is formed using the most recent data point (last row) of the DataFrame. The sequence is reshaped to match the input shape expected by the model.
7.Price prediction: The best model predicts the price based on the input sequence.
8.Deciphering the forecast: The predicted price is in the normalized scale. It is then transformed back to the original price scale using the inverse_transform method of the scaler.
9.Printing the forecast: The forecasted price for the next 5 minutes is printed.
In summary, the model is trained using historical Bitcoin price and volume data with an LSTM neural network. The trained model is then used to predict the price for the next 5 minutes based on the most recent data obtained from the Binance API.

After we have trained our model, we move on to BTCUSDT_TEST.py.
In this code, I also used the Telegram API where I created a bot for myself for convenience.  This part, based on the trained model, makes a prediction, and also compares whether the prediction was successful or not.

Thank you for your attention, I will be glad to any assessment, as well as advice!
