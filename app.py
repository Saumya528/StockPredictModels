import base64
import io
from flask import Flask, jsonify, request,render_template
from nsepython import *
import plotly.graph_objects as go
from datetime import datetime,timedelta
from DefineIndicators import fetch_indicators


# Initialize Flask app
app = Flask(__name__)

# Home route to render a basic HTML page or message
@app.route('/')
def home():
    return render_template('index.html')  # This serves the index.html from templates folder


@app.route('/stock-price', methods=['GET'])
def get_stock_price():
    """Endpoint to get the price of a stock."""
    # Get stock symbol from query parameters
    stock_symbol = request.args.get('symbol')
    
    if not stock_symbol:
        return jsonify({"error": "Stock symbol is required"}), 400

    try:
        # Fetch stock details
        stock_data = nse_quote_ltp(stock_symbol.upper())
        
        if not stock_data:
            return jsonify({"error": "Stock symbol not found"}), 404
        
        # Extract price information
        price = stock_data
        return jsonify({"symbol": stock_symbol.upper(), "price": price})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Flask code: Ensure correct JSON response structure

@app.route('/stock-chart', methods=['GET'])
def get_stock_chart():
    """Endpoint to get the candlestick chart and volume of a stock."""
    stock_symbol = request.args.get('symbol')
    
    if not stock_symbol:
        return jsonify({"error": "Stock symbol is required"}), 400

    try:
        end_date = datetime.now().strftime("%d-%m-%Y")
        start_date = (datetime.now() - timedelta(days=365)).strftime("%d-%m-%Y")
        series = "EQ"
        
        # Fetch stock data
        df = equity_history(stock_symbol.upper(), series, start_date, end_date)

        if df.empty:
            return jsonify({"error": "No data found for the stock symbol"}), 404

        # Create Plotly figure
        fig = go.Figure(data=[
            go.Candlestick(
                x=df['CH_TIMESTAMP'],
                open=df['CH_OPENING_PRICE'],
                high=df['CH_TRADE_HIGH_PRICE'],
                low=df['CH_TRADE_LOW_PRICE'],
                close=df['CH_CLOSING_PRICE'],
                name='Price'
            ),
            go.Bar(
                x=df['CH_TIMESTAMP'],
                y=df['CH_TOT_TRADED_QTY'],
                name='Volume',
                marker=dict(color='blue'),
                opacity=0.3
            )
        ])

        fig.update_layout(
            title=f'Candlestick Chart with Volume: {stock_symbol.upper()}',
            xaxis_title='Date',
            yaxis_title='Price',
            yaxis2=dict(
                title='Volume',
                overlaying='y',
                side='right',
                showgrid=False
            ),
            xaxis_rangeslider_visible=False
        )

        # Convert Plotly figure to JSON
        fig_json = json.loads(fig.to_json())
        return jsonify(fig_json)  # Return JSON object
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stock-indicator-chart', methods=['POST'])
def stock_chart():
    try:
        # Parse user input from the POST request
        request_data = request.json
        stock_symbol = request_data.get('symbol')
        strdt=datetime.strptime(request_data.get('start_date'),"%d-%m-%Y")
        endt=datetime.strfpime(request_data.get('end_date',"%d-%m-%Y"))
        #start_date = datetime.strptime(strdt,"%d-%m-%Y")
        #end_date = datetime.strfpime(endt,"%d-%m-%Y")

        start_date=datetime.strftime("'%Y-%m-%d'")
        end_date=datetime.strftime("'%Y-%m-%d'")
        

        if not stock_symbol or not start_date or not end_date:
            return jsonify({"error": "Symbol, start_date, and end_date are required."}), 400

        # Calculate all indicators
        indicators = ["SMA", "EMA", "RSI", "Bollinger", "MACD", "OBV"]
        result = fetch_indicators(stock_symbol.upper(), start_date, end_date, indicators)

        if "error" in result:
            return jsonify({"error": result["error"]}), 404

        # Create Plotly figure
        fig = go.Figure()

        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=result['dates'],
            open=result['open'],
            high=result['high'],
            low=result['low'],
            close=result['close'],
            name='Candlestick',
            hovertemplate=(
                '<b>Date</b>: %{x}<br>'
                '<b>Open</b>: %{open}<br>'
                '<b>High</b>: %{high}<br>'
                '<b>Low</b>: %{low}<br>'
                '<b>Close</b>: %{close}<extra></extra>'
            )
        ))

        # Add all indicators as separate traces
        for indicator in indicators:
            if indicator in result:
                fig.add_trace(go.Scatter(
                    x=result['dates'],
                    y=result[indicator],
                    mode='lines',
                    name=indicator,
                    hovertemplate=(
                        f'<b>Date</b>: %{x}<br>'
                        f'<b>{indicator} Value</b>: %{y}<br>'
                        '<extra></extra>'
                    )
                ))

        # Configure layout
        fig.update_layout(
            title=f"Stock Chart for {stock_symbol.upper()} with Indicators",
            xaxis_title="Date",
            yaxis_title="Price/Indicator Value",
            legend=dict(orientation="h"),
            height=700
        )

        # Convert Plotly figure to JSON
        fig_json = json.loads(fig.to_json())
        return jsonify(fig_json)  # Return JSON object
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

