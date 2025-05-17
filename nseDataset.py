import sys
print("Python Executable:", sys.executable)
print("IPython Detected:", 'IPython' in sys.modules)


import pandas as pd
import io
import os
import plotly.graph_objects as go
from nsepy import get_history
from datetime import datetime, timedelta

# === Folder Setup ===
SAVE_FOLDER = "Data_downloaded"
os.makedirs(SAVE_FOLDER, exist_ok=True)

# === Function to Fetch Stock Data ===
def get_stock_data(stock_symbol):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365*2)

    df = get_history(symbol=stock_symbol, start=start_date, end=end_date)
    df = df.reset_index()
    df = df[['Date', 'Open', 'High', 'Low', 'Close']]
    return df

# === Function to Save as Excel ===
def save_to_excel(df, filepath):
    output = io.BytesIO()
    df.to_excel(output, index=True, engine='openpyxl')
    output.seek(0)
    with open(filepath, 'wb') as f:
        f.write(output.read())

# === Delete File by Name ===
def delete_file_by_name(filename):
    filepath = os.path.join(SAVE_FOLDER, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"✅ File '{filename}' deleted.")
    else:
        print(f"⚠️ File '{filename}' not found in '{SAVE_FOLDER}'.")

# === Optional Deletion Before Download ===
def check_and_delete_existing_file():
    delete_existing = input("Do you want to delete any existing file? (y/n): ").strip().lower()
    if delete_existing == 'y':
        print(f"\n📁 Available files in '{SAVE_FOLDER}':")
        files = os.listdir(SAVE_FOLDER)
        if not files:
            print("No files found.")
            return
        for file in files:
            print(f" - {file}")
        filename_to_delete = input("\nEnter the filename to delete (with extension): ").strip()
        delete_file_by_name(filename_to_delete)

# === Main Workflow ===
def main():
    check_and_delete_existing_file()

    stock_symbol = input("\nEnter the stock symbol to download (e.g., TCS, RELIANCE): ").strip().upper()

    try:
        stock_data = get_stock_data(stock_symbol)
        print("\n✅ Downloaded Data (first 5 rows):")
        print(stock_data.head())

        # Add daily return column
        stock_data['Daily Return'] = stock_data['Close'].pct_change()

        # Save to Excel
        filename = f"{stock_symbol}_stock_data_nse.xlsx"
        filepath = os.path.join(SAVE_FOLDER, filename)
        save_to_excel(stock_data, filepath)
        print(f"\n📊 Data saved to: {filepath}")

        # Ask to delete after save
        delete_latest = input(f"\nDo you want to delete the newly saved file '{filename}'? (y/n): ").strip().lower()
        if delete_latest == 'y':
            delete_file_by_name(filename)
        else:
            print("📁 File kept.")

    except Exception as e:
        print(f"\n❌ Error: {e}")

# === Entry Point ===
if __name__ == "__main__":
    main()
