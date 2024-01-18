import yfinance as yf
import pandas as pd

def fetch_data(tickers, period="1mo", interval="1d"):
    data = yf.download(tickers, period=period, interval=interval)
    return data['Close']

def analyze_stocks(data):
    # Example analysis: Calculate percentage change
    perf = data.pct_change().mean() * 100
    vol = data.pct_change().std() * 100
    return pd.DataFrame({'Performance': perf, 'Volatility': vol})

def select_top_stocks(analysis, top_n=5):
    # Filter based on criteria (example: performance and volatility)
    filtered = analysis[(analysis['Performance'] > 0) & (analysis['Volatility'] > 2)]
    return filtered.sort_values(by='Performance', ascending=False).head(top_n)

# List of stock tickers
tickers = ['HDFCBANK.NS',
           'RELIANCE.NS',
           'ICICIBANK.NS',
           'INFY.NS',
           'LT.NS',
           'ITC.NS',
           'TCS.NS',
           'AXISBANK.NS',
           'KOTAKBANK.NS',
           'BHARTIARTL.NS',
           'HINDUNILVR.NS',
           'BAJFINANCE.NS',
           'M&M.NS',
           'HCLTECH.NS',
           'TITAN.NS',
           'ASIANPAINT.NS',
           'NTPC.NS',
           'TATAMOTORS.NS',
           'MARUTI.NS',
           'SUNPHARMA.NS',
           'ULTRACEMCO.NS',
           'TATASTEEL.NS',
           'POWERGRID.NS',
           'INDUSINDBK.NS',
           'NESTLEIND.NS',
           'HINDALCO.NS',
           'COALINDIA.NS',
           'JSWSTEEL.NS',
           'ONGC.NS',
           'TECHM.NS',
           'GRASIM.NS',
           'BAJAJ-AUTO.NS',
           'ADANIENT.NS',
           'ADANIPORTS.NS',
           'DRREDDY.NS',
           'HDFCLIFE.NS',
           'WIPRO.NS',
           'CIPLA.NS',
           'TATACONSUM.NS',
           'SBILIFE.NS',
           'BRITANNIA.NS',
           ] # Modify as needed

# Fetch stock data
stock_data = fetch_data(tickers)

# Analyze the stocks
analysis = analyze_stocks(stock_data)

# Select top performing stocks
top_stocks = select_top_stocks(analysis)
print(top_stocks)

