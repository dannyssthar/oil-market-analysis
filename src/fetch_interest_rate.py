# src/fetch_interest_rate.py

import os
from dotenv import load_dotenv
from fredapi import Fred
import pandas as pd

# Load environment variables
load_dotenv()

# Connect to FRED
fred = Fred(api_key=os.getenv("FRED_API_KEY"))

def fetch_interest_rate():
    """
    Fetch US 10-Year Treasury Yield from FRED
    Symbol: DGS10

    Returns
    -------
    pandas.Series
        Daily interest rate series
    """

    rate = fred.get_series("DGS10")

    rate = rate.rename("InterestRate")
    rate.index = pd.to_datetime(rate.index)

    return rate


if __name__ == "__main__":
    data = fetch_interest_rate()

    # Save locally
    data.to_csv("data/interest_rate.csv")

    print("Interest rate data saved.")
