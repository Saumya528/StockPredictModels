import base64
import io
from flask import Flask, jsonify, request, render_template
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
from DefineIndicators import fetch_indicators
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from flask import Response
import json
import matplotlib.pyplot as plt
import mplfinance as mpf



# Initialize Flask app
app = Flask(__name__)

# Global variables for the model
model  = None
scaler  = None
imputer = None

def calculate_fibonacci_levels(df):
    """
    Calculate Fibonacci retracement and extension levels.
    Adds these as new columns in the DataFrame.
    """
    high_price = df['High'].max()
    low_price = df['Low'].min()
    
    diff = high_price - low_price
    fib_levels = {
        'Fibonacci_23_6': high_price - 0.236 * diff,
        'Fibonacci_38_2': high_price - 0.382 * diff,
        'Fibonacci_50_0': high_price - 0.5 * diff,
        'Fibonacci_61_8': high_price - 0.618 * diff
    }

    fib_expansions = {
        'Fibonacci_1618': high_price + 1.618 * diff,
        'Fibonacci_2618': high_price + 2.618 * diff
    }

    # Add these levels to the dataframe
    for key, value in fib_levels.items():
        df[key] = value
    for key, value in fib_expansions.items():
        df[key] = value

    return df

def prepare_data(stock_symbol):
    try:
        # Fetch historical stock data using yfinance
        df = yf.download(stock_symbol, period="2y", interval="1d")
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

        # Validate data presence
        if df.empty:
            raise ValueError(f"No data retrieved for {stock_symbol}.")

        # Calculate moving averages
        df['Returns'] = df['Close'].pct_change()
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()

        # Add Fibonacci levels
        df = calculate_fibonacci_levels(df)

        # Predict next day's movement (1 if price goes up, 0 if price goes down)
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

        # Handle missing values using forward and backward fill
        df.fillna(method='bfill', inplace=True)
        df.fillna(method='ffill', inplace=True)

        if len(df) < 20:
            raise ValueError(f"Insufficient data points ({len(df)}) for {stock_symbol}.")

        return df

    except Exception as e:
        print(f"Error preparing data for {stock_symbol}: {e}")
        raise

def train_logistic_regression_model(df):
    # Feature columns (including 'Returns', 'SMA_5', 'SMA_20', and Fibonacci levels)
    features = ['Returns', 'SMA_5', 'SMA_20', 'Fibonacci_23_6', 'Fibonacci_38_2', 'Fibonacci_50_0', 'Fibonacci_61_8', 'Fibonacci_1618', 'Fibonacci_2618']
    X = df[features]
    y = df['Target']

    # Handle missing values using SimpleImputer to replace NaN with the mean value of the column
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)  # Impute the missing values in the features

    # Split the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

    # Standardize the features (important for Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fitting and scaling on training data
    X_test_scaled = scaler.transform(X_test)  # Using the same scaler for test data
    
    print(X_train)
    print(X_train_scaled)

    # Initialize and train the Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test_scaled)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

    return model, scaler, imputer, accuracy

def predict_next_day_logistic(stock_symbol, model, scaler, imputer):
    try:
        # Prepare data and fetch the latest row (latest trading day)
        df = prepare_data(stock_symbol)
        
        # Get the latest data (Returns, SMA_5, SMA_20, Fibonacci levels)
        latest_data = df.iloc[-1][['Returns', 'SMA_5', 'SMA_20', 'Fibonacci_23_6', 'Fibonacci_38_2', 'Fibonacci_50_0', 'Fibonacci_61_8', 'Fibonacci_1618', 'Fibonacci_2618']].values.reshape(1, -1)

        # Impute missing values (if any) before scaling
        latest_data_imputed = imputer.transform(latest_data)

        # Scale the latest data using the previously fitted scaler
        latest_data_scaled = scaler.transform(latest_data_imputed)

        # Predict the next day's movement (1 = up, 0 = down)
        prediction = model.predict(latest_data_scaled)

        # Convert prediction to a native Python type (scalar)
        prediction = int(prediction[0])  # Convert the prediction to a scalar (1 or 0)

        # Get the latest closing price for the stock
        latest_close = df['Close'].iloc[-1]

        # Calculate the approximate next day's price based on predicted movement
        if prediction == 1:
            # Price is expected to go up
            predicted_price = latest_close * (1 + df['Returns'].iloc[-1])  # Correctly apply return for upward prediction
        else:
            # Price is expected to go down
            predicted_price = latest_close * (1 - df['Returns'].iloc[-1])  # Correctly apply return for downward prediction

        # Return the prediction: movement (Up/Down) and predicted price for the next day
        print(f"Prediction: The stock is predicted to go {'Up' if prediction == 1 else 'Down'}")
        print(f"Predicted Price: {round(predicted_price, 2)}")

        return ('Up' if prediction == 1 else 'Down'), round(predicted_price, 2)

    except Exception as e:
        print(f"Error in predicting next day's movement and price: {str(e)}")
        return None, None  # Return None in case of error


def train_svm_model(df):
    """
    Train a Support Vector Machine (SVM) model using features like 'Returns', 'SMA', and 'Fibonacci levels'.
    """
    # Feature columns (including 'Returns', 'SMA_5', 'SMA_20', and Fibonacci levels)
    features = ['Returns', 'SMA_5', 'SMA_20', 'Fibonacci_23_6', 'Fibonacci_38_2', 'Fibonacci_50_0', 'Fibonacci_61_8', 'Fibonacci_1618', 'Fibonacci_2618']
    X = df[features]
    y = df['Target']  # Target: Up (1) or Down (0) for the stock price movement

    # Handle missing values using SimpleImputer to replace NaN with the mean value of the column
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)  # Impute missing values in the features

    # Split the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

    # Standardize the features (important for SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit and scale on the training data
    X_test_scaled = scaler.transform(X_test)  # Use the same scaler for the test data

    # Initialize and train the SVM model
    model = SVC(kernel='linear', random_state=42)  # Using a linear kernel, but other kernels can be explored
    model.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test_scaled)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

    return model, scaler, imputer, accuracy

# Example usage:
# Assuming df is your DataFrame containing stock data (including features and target)
# model, scaler, imputer, accuracy = train_svm_model(df)


# Function to predict the next day's stock movement and price using the trained SVM model
def predict_next_day_svm(stock_symbol, model, scaler, imputer):
    try:
        # Prepare data and fetch the latest row (latest trading day)
        df = prepare_data(stock_symbol)
        
        # Get the latest data (Returns, SMA_5, SMA_20, Fibonacci levels)
        latest_data = df.iloc[-1][['Returns', 'SMA_5', 'SMA_20', 'Fibonacci_23_6', 'Fibonacci_38_2', 'Fibonacci_50_0', 'Fibonacci_61_8', 'Fibonacci_1618', 'Fibonacci_2618']].values.reshape(1, -1)

        # Impute missing values (if any) before scaling
        latest_data_imputed = imputer.transform(latest_data)

        # Scale the latest data using the previously fitted scaler
        latest_data_scaled = scaler.transform(latest_data_imputed)

        # Predict the next day's movement (1 = up, 0 = down)
        prediction = model.predict(latest_data_scaled)

        # Convert prediction to a native Python type (scalar)
        prediction = int(prediction[0])  # Convert the prediction to a scalar (1 or 0)

        # Get the latest closing price for the stock
        latest_close = df['Close'].iloc[-1]

        # Calculate the approximate next day's price based on predicted movement
        if prediction == 1:
            # Price is expected to go up
            predicted_price = latest_close * (1 + df['Returns'].iloc[-1])  # Correctly apply return for upward prediction
        else:
            # Price is expected to go down
            predicted_price = latest_close * (1 - df['Returns'].iloc[-1])  # Correctly apply return for downward prediction

        # Return the prediction: movement (Up/Down) and predicted price for the next day
        print(f"Prediction: The stock is predicted to go {'Up' if prediction == 1 else 'Down'}")
        print(f"Predicted Price: {round(predicted_price, 2)}")

        return ('Up' if prediction == 1 else 'Down'), round(predicted_price, 2)

    except Exception as e:
        print(f"Error in predicting next day's movement and price: {str(e)}")
        return None, None  # Return None in case of error

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/stock-price', methods=['GET'])
def get_stock_price():
    stock_symbol = request.args.get('symbol')

    if not stock_symbol:
        return jsonify({"error": "Stock symbol is required"}), 400

    try:
        stock_data = yf.Ticker(stock_symbol)
        info = stock_data.info
        price = info.get('currentPrice') or info.get('regularMarketPrice')

        if not price:
            raise ValueError("Stock price not available. Please check the symbol.")

        return jsonify({
            "symbol": stock_symbol.upper(),
            "price": round(price, 2)
        })

    except Exception as e:
        return jsonify({"error": f"❌ Failed to fetch price for {stock_symbol}: {str(e)}"}), 500


@app.route('/stock-chart', methods=['GET'])
def get_stock_chart():
    stock_symbol = request.args.get('symbol')

    if not stock_symbol:
        return jsonify({"error": "Stock symbol is required"}), 400

    try:
        # Download the stock data using yfinance
        df = yf.download(stock_symbol, period="1y", interval="1d")

        # Flatten the MultiIndex columns
        df.columns = ['_'.join(col).strip() for col in df.columns.values]

        # Log the DataFrame for debugging (check if everything is fine)
        print(f"Data for {stock_symbol}:")
        print(df.head())  # Print the first few rows

        # Ensure the DataFrame contains only the necessary columns
        df = df[['Open_' + stock_symbol.upper(), 'High_' + stock_symbol.upper(), 'Low_' + stock_symbol.upper(), 'Close_' + stock_symbol.upper(), 'Volume_' + stock_symbol.upper()]]


        # Ensure the DataFrame contains only the necessary columns
        #df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

        # Ensure the index is a DatetimeIndex
        df.index.name = 'Date'  # Name the index as 'Date'
        df.index = pd.to_datetime(df.index)  # Convert index to datetime if it's not already

        # Ensure all columns are numeric (float or int)
        df['Open'] = pd.to_numeric(df['Open_' + stock_symbol.upper()], errors='coerce')
        df['High'] = pd.to_numeric(df['High_' + stock_symbol.upper()], errors='coerce')
        df['Low'] = pd.to_numeric(df['Low_' + stock_symbol.upper()], errors='coerce')
        df['Close'] = pd.to_numeric(df['Close_' + stock_symbol.upper()], errors='coerce')
        df['Volume'] = pd.to_numeric(df['Volume_' + stock_symbol.upper()], errors='coerce')

        # Check for empty data
        if df.empty:
            return jsonify({"error": "No data found for the stock symbol"}), 404

        # Handle any NaN values by filling them (forward fill or backward fill)
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)

        # Log the DataFrame for debugging (check if everything is fine)
        print(f"Data for {stock_symbol}:")
        print(df.head())  # Print the first few rows

        # Ensure x (Date) is a list (series can be passed as well)
        x_values = df.index.tolist()
        

        fig = go.Figure(data=[  
            go.Candlestick(
                x=x_values,  # Ensure Date is used for x-axis (DatetimeIndex)
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price'
            ),
            go.Bar(
                x=x_values,  # Ensure Date is used for x-axis (DatetimeIndex)
                y=df['Volume'],
                name='Volume',
                marker=dict(color='blue'),
                opacity=0.3,
                yaxis='y2'  # Secondary y-axis for volume
            )
        ])

        fig.update_layout(
            title=f'Candlestick Chart with Volume: {stock_symbol.upper()}',
            xaxis_title='Date',
            yaxis_title='Price',
            yaxis2=dict(title='Volume', overlaying='y', side='right', showgrid=False),
            xaxis_rangeslider_visible=False
        )

        # Return the chart in JSON format
        fig_json = json.loads(fig.to_json())
        return jsonify(fig_json)  # Return JSON object

    except Exception as e:
        return jsonify({"error": f"❌ Error generating chart: {str(e)}"}), 500
    
# Route for training the Logistic Regression model
@app.route('/train-logistic', methods=['POST'])
def train_logistic_regression_model_route():
    try:
        data = request.json
        stock_symbol = data.get('symbol')

        if not stock_symbol:
            return jsonify({"error": "Missing required parameters"}), 400

        # Prepare data and train the model
        df = prepare_data(stock_symbol)
        model, scaler, imputer, accuracy = train_logistic_regression_model(df)

        # Save the model and scaler for future use (if needed)
        global global_model, global_scaler, global_imputer
        global_model = model
        global_scaler = scaler
        global_imputer = imputer

        return jsonify({"message": "Logistic Regression Model trained successfully", "accuracy": accuracy})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route for training the SVM model
@app.route('/train-svm', methods=['POST'])
def train_svm_model_route():
    try:
        data = request.json
        stock_symbol = data.get('symbol')

        if not stock_symbol:
            return jsonify({"error": "Missing required parameters"}), 400

        # Prepare data and train the model
        df = prepare_data(stock_symbol)
        model, scaler, imputer, accuracy = train_svm_model(df)

        # Save the model and scaler for future use (if needed)
        global global_model, global_scaler, global_imputer
        global_model = model
        global_scaler = scaler
        global_imputer = imputer

        return jsonify({"message": "SVM Model trained successfully", "accuracy": accuracy})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict-next-day', methods=['POST'])
def predict_next_day_route():
    try:
        data = request.json
        stock_symbol = data.get('symbol')

        if not stock_symbol:
            return jsonify({"error": "Missing required parameters"}), 400

        # Predict the next day's movement and price
        if 'global_model' not in globals() or 'global_scaler' not in globals() or 'global_imputer' not in globals():
            return jsonify({"error": "Model has not been trained yet. Please train the model first."}), 400

        # Call the prediction function with model, scaler, and imputer
        prediction_logistic, predicted_price_logistic = predict_next_day_logistic(stock_symbol, global_model, global_scaler, global_imputer)

        prediction_svm, predicted_price_svm = predict_next_day_svm(stock_symbol, global_model, global_scaler, global_imputer)

        # Ensure the prediction and predicted price are serializable
        if prediction_svm is None or predicted_price_svm is None:
            return jsonify({"error": "Error predicting the next day's stock movement."}), 500
        if prediction_logistic is None or predicted_price_logistic is None:
            return jsonify({"error": "Error predicting the next day's stock movement."}), 500

        # Convert prediction and predicted_price to native Python types if necessary
        prediction_logistic = str(prediction_logistic)  # Ensure it's a string ("Up" or "Down")
        predicted_price_logistic = float(predicted_price_logistic)  # Ensure it's a float
        prediction_svm = str(prediction_svm)  # Ensure it's a string ("Up" or "Down")
        predicted_price_svm = float(predicted_price_svm)  # Ensure it's a float

        return jsonify({
            "predicted_movement_logistic": prediction_logistic,
            "predicted_price_logistic": round(predicted_price_logistic, 2),
            "predicted_movement_svm": prediction_svm,
            "predicted_price_svm": round(predicted_price_svm, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
