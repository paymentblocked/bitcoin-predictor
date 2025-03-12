import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Fetch data from CoinGecko API
def fetch_coingecko_data(coin_id="bitcoin", vs_currency="usd", days="1"):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
    params = {
        "vs_currency": vs_currency,  # Currency to compare against
        "days": days                # Number of days of data to fetch
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        raise Exception(f"Failed to fetch data: {response.status_code}")

# Preprocess the data
def preprocess_data(data):
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")  # Convert timestamp to datetime

    # Use the 'close' price as the target variable
    prices = df['close'].values.reshape(-1, 1)

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)
    return df, prices, scaled_prices, scaler

# Create sequences for the LSTM model
def create_dataset(data, look_back=60):
    X, y = [], []
    for i in range(len(data) - look_back - 1):
        X.append(data[i:(i + look_back)])  # Past 'look_back' values
        y.append(data[i + look_back])      # Next value
    return np.array(X), np.array(y)

#Build LSTM model
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def main():
    # Fetch data
    data = fetch_coingecko_data(coin_id="bitcoin", vs_currency="usd", days="1")

    # Preprocess data
    df, prices, scaled_prices, scaler = preprocess_data(data)

    # Create sequences
    look_back = 30
    X, y = create_dataset(scaled_prices, look_back)

    # Reshape X to be [samples, timesteps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Split the data into training and testing sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build the model
    model = build_model(input_shape=(X_train.shape[1], 1))

    # Train the model
    model.fit(X_train, y_train, batch_size=1, epochs=10, verbose=1)

    # Evaluate the model
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
    print(f'Root Mean Squared Error: {rmse}')

    # Predict the next movement
    last_sequence = scaled_prices[-look_back:]
    last_sequence = np.reshape(last_sequence, (1, look_back, 1))
    next_price = model.predict(last_sequence)
    next_price = scaler.inverse_transform(next_price)
    print(f'Predicted next price: {next_price[0][0]}')

    # Determine the direction of the next movement
    current_price = prices[-1][0]
    predicted_price = next_price[0][0]
    if predicted_price > current_price:
        print("Predicted movement: Up")
    else:
        print("Predicted movement: Down")


if __name__ == "__main__":
    main()