# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# from keras.models import Sequential
# from keras.layers import LSTM, Dense, Dropout
# from keras.callbacks import EarlyStopping

# # Load and preprocess data
# data = pd.read_csv('reliance.csv', parse_dates=['Date'])
# data.set_index('Date', inplace=True)

# # Check if data is loaded correctly
# print("Data shape after loading:", data.shape)

# # Feature Engineering
# data['MA50'] = data['Close'].rolling(window=50).mean()  # 50 days moving average
# data['MA200'] = data['Close'].rolling(window=200).mean()  # 200 days moving average
# data = data.dropna()  # Dropping NA values after rolling mean calculation

# # Check data shape after dropna
# print("Data shape after dropna:", data.shape)

# # Select features and target
# features = data[['Open', 'High', 'Low', 'Volume', 'MA50', 'MA200']]
# target = data['Close']

# # Scale the features and target
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_features = scaler.fit_transform(features)
# scaled_target = scaler.fit_transform(target.values.reshape(-1, 1))

# # Create dataset for LSTM
# def create_dataset(X, y, time_steps):
#     Xs, ys = [], []
#     for i in range(len(X) - time_steps):
#         v = X[i:(i + time_steps)]
#         Xs.append(v)
#         ys.append(y[i + time_steps])
#     return np.array(Xs), np.array(ys)

# X, y = create_dataset(scaled_features, scaled_target, 40)

# # Check if dataset creation is successful
# print("Shape of X, y:", X.shape, y.shape)

# # Verify that X and y are not empty
# if X.size == 0 or y.size == 0:
#     raise ValueError("X or y arrays are empty. Check the time_steps or data preprocessing steps.")

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# # Build the LSTM model
# model = Sequential()
# model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
# model.add(Dropout(0.2))
# model.add(LSTM(units=100, return_sequences=False))
# model.add(Dropout(0.2))
# model.add(Dense(units=1))

# # Compile and train the model
# model.compile(optimizer='adam', loss='mean_squared_error')
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping], shuffle=False)

# # Predict and inverse transform to original scale
# y_pred = model.predict(X_test)
# y_pred = scaler.inverse_transform(y_pred)

# # Plot the results
# plt.figure(figsize=(12, 6))
# plt.plot(data.index[-len(y_test):], scaler.inverse_transform(y_test.reshape(-1, 1)), label='True')
# plt.plot(data.index[-len(y_test):], y_pred, label='Predicted')
# plt.xlabel('Time')
# plt.ylabel('Stock Price')
# plt.title('Advanced ICICI Stock Price Prediction')
# plt.legend()
# plt.show()
