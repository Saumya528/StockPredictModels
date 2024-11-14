import pandas as pd
from nsepython import equity_history

def fetch_and_prepare_data(symbol, start_date, end_date):
    series = "EQ"
    df = equity_history(symbol, series, start_date, end_date)

    # Prepare DataFrame
    df['close'] = pd.to_numeric(df['CH_CLOSING_PRICE'], errors='coerce')
    df['volume'] = pd.to_numeric(df['CH_TOT_TRADED_QTY'], errors='coerce')
    df['open'] = pd.to_numeric(df['CH_OPENING_PRICE'], errors='coerce')
    df['high'] = pd.to_numeric(df['CH_TRADE_HIGH_PRICE'], errors='coerce')
    df['low'] = pd.to_numeric(df['CH_TRADE_LOW_PRICE'], errors='coerce')
    df['date'] = pd.to_datetime(df['CH_TIMESTAMP'])

    # Drop rows with missing data
    df.dropna(subset=['close', 'volume'], inplace=True)
    return df

def calculate_sma(df):
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    return df

def calculate_ema(df):
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    return df

def calculate_rsi(df):
    delta = df['close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def calculate_bollinger_bands(df):
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['std_dev'] = df['close'].rolling(window=20).std()
    df['Upper_Band'] = df['SMA_20'] + (2 * df['std_dev'])
    df['Lower_Band'] = df['SMA_20'] - (2 * df['std_dev'])
    return df

def calculate_macd(df):
    df['MACD'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

def calculate_obv(df):
    df['OBV'] = df.apply(
        lambda row: row['volume'] if row['close'] > row['close'].shift(1)
        else -row['volume'] if row['close'] < row['close'].shift(1)
        else 0, axis=1
    ).cumsum()
    return df

def fetch_indicators(symbol, start_date, end_date, indicators):
    df = fetch_and_prepare_data(symbol, start_date, end_date)
    if df.empty:
        return {"error": "No data available for the given symbol and dates."}

    indicator_functions = {
        "SMA": calculate_sma,
        "EMA": calculate_ema,
        "RSI": calculate_rsi,
        "Bollinger": calculate_bollinger_bands,
        "MACD": calculate_macd,
        "OBV": calculate_obv,
    }

    for indicator in indicators:
        if indicator in indicator_functions:
            df = indicator_functions[indicator](df)

    results = {
        "dates": df['date'].dt.strftime('%Y-%m-%d').tolist(),
        "close": df['close'].tolist(),
        "open": df['open'].tolist(),
        "high": df['high'].tolist(),
        "low": df['low'].tolist(),
        "volume": df['volume'].tolist(),
    }

    # Add indicators to results
    for indicator in indicators:
        if indicator in df.columns:
            results[indicator] = df[indicator].tolist()

    # Replace NaN with None for JSON serialization
    results = {key: [None if pd.isna(x) else x for x in value] for key, value in results.items()}
    return results
