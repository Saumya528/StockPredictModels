import requests
import pandas as pd
from datetime import datetime, timedelta
import os


# Folder to save Excel files
SAVE_FOLDER = "Data_downloaded"

# Ensure the folder exists
os.makedirs(SAVE_FOLDER, exist_ok=True)

current_year = datetime.now().year
save_path = os.path.join(SAVE_FOLDER, f"market_closed_days_{current_year}.csv")

def fetch_nse_closed_days_for_current_year():
    current_year = datetime.now().year

    # === Step 1: Fetch NSE Holidays from Upstox API ===
    url = "https://api.upstox.com/v2/market/holidays"
    headers = {
        "Accept": "application/json"
    }

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch holidays. Status code: {response.status_code}")

    holidays_json = response.json().get("data", [])
    holiday_df = pd.DataFrame(holidays_json)

    # Convert date column to datetime
    holiday_df["date"] = pd.to_datetime(holiday_df["date"])
    holiday_df = holiday_df[holiday_df["date"].dt.year == current_year]

    # === Step 2: Add all Saturdays and Sundays ===
    start_date = datetime(current_year, 1, 1)
    end_date = datetime(current_year, 12, 31)
    all_dates = pd.date_range(start=start_date, end=end_date)

    weekend_dates = [d for d in all_dates if d.weekday() >= 5]  # Saturday=5, Sunday=6
    weekend_df = pd.DataFrame({
        "date": weekend_dates,
        "description": ["Weekend"] * len(weekend_dates),
        "holiday_type": ["WEEKEND"] * len(weekend_dates),
        "closed_exchanges": [["NSE", "BSE"]] * len(weekend_dates),
        "open_exchanges": [[]] * len(weekend_dates)
    })

    # === Step 3: Combine and sort all closed days ===
    all_closed_df = pd.concat([holiday_df, weekend_df], ignore_index=True)
    all_closed_df = all_closed_df.sort_values("date").reset_index(drop=True)

    # ✅ Save to CSV
    all_closed_df.to_csv(save_path, index=False)
    print(f"✅ Market closed days saved to: {save_path}")

    return all_closed_df



fetch_nse_closed_days_for_current_year()
