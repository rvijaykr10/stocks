import yfinance as yf

def fetch_nse_data(ticker_symbol, start_date, end_date):
    # Format the ticker symbol for NSE by adding '.NS' at the end
    ticker_nse = f"{ticker_symbol}.NS"

    # Fetch data
    data = yf.download(ticker_nse, start=start_date, end=end_date)
    return data

# Example usage
ticker = "RELIANCE"  # Replace with the ticker symbol of your choice
start_date = "2024-01-01" #yyyy-mm-dd
end_date = "2024-01-15" #yyyy-mm-dd

# Fetching the data
stock_data = fetch_nse_data(ticker, start_date, end_date)

# Display the data
print(stock_data)