import yfinance as yf
import io
import os
import pandas as pd
from datetime import datetime, timedelta

def prepare_data(stock_symbol):
    try:
       # Fetch historical stock data using nsepython
        df = yf.download(stock_symbol, period="2y", interval="1d")
        df=df[['Open', 'High', 'Low', 'Close']]

        # Print the raw response for debugging
        print(f"Raw response for {stock_symbol}: {df}")

        # Validate data presence
        if df.empty:
            raise ValueError(f"⚠️ No data retrieved for {stock_symbol}")

        # Ensure 'CH_CLOSING_PRICE' exists, fallback to 'CH_CLOSE'
        if 'Close' in df.columns:
            df['Close'] = df['Close']
        elif 'CH_CLOSE' in df.columns:
            df['Close'] = df['CH_CLOSE']
        else:
            print(f"Available columns: {df.columns}")  # Debug available columns
            raise KeyError(f"⚠️ Column 'CH_CLOSING_PRICE' or 'CH_CLOSE' not found in data for {stock_symbol}")

        # Compute indicators
        df['Returns'] = df['Close'].pct_change()
        df['SMA_5'] = df['Close'].rolling(window=5, min_periods=1).mean()
        df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

        # Handle missing values using forward and backward fill
        df.fillna(method='bfill', inplace=True)
        df.fillna(method='ffill', inplace=True)

        # Ensure at least 20 valid data points
        if len(df) < 20:
            raise ValueError(f"⚠️ Insufficient data points ({len(df)}) for {stock_symbol} after preprocessing")

        return df

    except KeyError as ke:
        raise KeyError(f"❌ Missing essential column: {str(ke)}")
    except Exception as e:
        raise ValueError(f"❌ Error preparing data for {stock_symbol}: {str(e)}")

# Example Usage
symbol = "TCS.NS" 

df = prepare_data(symbol)
print(df.tail())  # Print last few rows for verification
