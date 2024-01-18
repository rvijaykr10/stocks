import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the historical stock data (CSV file)
data = pd.read_csv('nifty_50.csv')

# Prepare the data
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Use Close price as the target variable
X = data[['Open', 'High', 'Low', 'Volume']]  # Features
y = data['Close']  # Target

# Drop rows where any of the data in X or y is NaN
combined = pd.concat([X, y], axis=1)
combined_cleaned = combined.dropna()

# Separate the features and target variable after removing NaNs
X_cleaned = combined_cleaned[['Open', 'High', 'Low', 'Volume']]
y_cleaned = combined_cleaned['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=0)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')
