import yfinance as yf
import pandas as pd
import io
import os
import plotly.graph_objects as go

# Folder to save Excel files
SAVE_FOLDER = "Data_downloaded"

# Ensure the folder exists
os.makedirs(SAVE_FOLDER, exist_ok=True)

# Function to fetch stock data
def get_stock_data(stock_symbol):
    stock_data = yf.download(stock_symbol, period="2y", interval="1d")
    return stock_data

# Function to save data as Excel
def save_to_excel(df, filepath):
    output = io.BytesIO()
    df.to_excel(output, index=True, engine='openpyxl')
    output.seek(0)
    with open(filepath, 'wb') as f:
        f.write(output.read())

# Function to delete a specified file
def delete_file_by_name(filename):
    filepath = os.path.join(SAVE_FOLDER, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"✅ File '{filename}' deleted.")
    else:
        print(f"⚠️ File '{filename}' not found in '{SAVE_FOLDER}'.")

# Function to prompt user to delete a file before fetching data
def check_and_delete_existing_file():
    delete_existing = input("Do you want to delete any existing file? (y/n): ").strip().lower()
    if delete_existing == 'y':
        print(f"\nAvailable files in '{SAVE_FOLDER}':")
        files = os.listdir(SAVE_FOLDER)
        if not files:
            print("No files found.")
            return
        for file in files:
            print(f" - {file}")
        filename_to_delete = input("\nEnter the filename to delete (with extension): ").strip()
        delete_file_by_name(filename_to_delete)

# Function to plot the candlestick chart
def plot_candlestick_chart(df, stock_symbol):
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    )])

    fig.update_layout(
        title=f'Candlestick Chart: {stock_symbol.upper()}',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False
    )
    
    fig.show()

def main():
    # Optionally delete an existing file before continuing
    check_and_delete_existing_file()

    stock_symbol = input("\nEnter the stock symbol to download (e.g., AAPL, MSFT): ").strip().upper()

    try:
        # Fetch stock data
        stock_data = get_stock_data(stock_symbol)
        print("\nDownloaded Data (first 5 rows):")
        print(stock_data.head())
        print(stock_data.columns)
        stock_data['Date']=stock_data.index
        df=stock_data['Close'].pct_change()
        print(df.head())
        print(stock_data['Date'])
        print(stock_data.index)


        # Save the file
        filename = f"{stock_symbol}_stock_data.xlsx"
        filepath = os.path.join(SAVE_FOLDER, filename)
        save_to_excel(stock_data, filepath)
        print(f"\n✅ Data saved to: {filepath}")

        # Plot the candlestick chart
        #plot_candlestick_chart(stock_data, stock_symbol)

        # Ask if user wants to delete this file
        delete_latest = input(f"\nDo you want to delete the newly saved file '{filename}'? (y/n): ").strip().lower()
        if delete_latest == 'y':
            delete_file_by_name(filename)
        else:
            print("File kept.")

    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
