# ================================
# Oil Market Macro Relationship Analysis
# Author: Daniel
# ================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import statsmodels.api as sm

plt.style.use("default")

# Ensure output directory exists
os.makedirs("figures", exist_ok=True)


# ------------------------------
# 1. Data Download
# ------------------------------

def fetch_price_series(ticker, start):
    """
    Robust downloader that handles:
    - MultiIndex columns
    - Missing Adj Close
    - Empty downloads
    """
    df = yf.download(ticker, start=start, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for ticker: {ticker}")

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Prefer Adj Close, fallback to Close
    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"

    return df[price_col]


def download_data(start="2015-01-01"):
    tickers = {
        "Oil": "CL=F",
        "USD": "DX-Y.NYB",
        "SP500": "^GSPC"
    }

    data = {}

    for name, ticker in tickers.items():
        print(f"Downloading {name}...")
        data[name] = fetch_price_series(ticker, start)

    combined = pd.DataFrame(data)
    combined.dropna(inplace=True)

    return combined


# ------------------------------
# 2. Transformations
# ------------------------------

def compute_returns(df):
    returns = np.log(df / df.shift(1))
    returns.dropna(inplace=True)
    return returns


# ------------------------------
# 3. Visualization
# ------------------------------

def plot_prices(df):
    plt.figure(figsize=(12,6))
    df.plot()
    plt.title("Asset Price Levels")
    plt.tight_layout()
    plt.savefig("figures/price_levels.png")
    plt.close()


def plot_returns(df):
    plt.figure(figsize=(12,6))
    df.plot()
    plt.title("Log Returns")
    plt.tight_layout()
    plt.savefig("figures/log_returns.png")
    plt.close()


# ------------------------------
# 4. Regression Model
# ------------------------------

def regression_analysis(returns):

    y = returns["Oil"]
    X = returns[["USD", "SP500"]]

    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    with open("figures/regression_summary.txt", "w") as f:
        f.write(model.summary().as_text())

    return model


# ------------------------------
# 5. Execution Pipeline
# ------------------------------

def main():

    prices = download_data()
    returns = compute_returns(prices)

    plot_prices(prices)
    plot_returns(returns)

    model = regression_analysis(returns)

    print(model.summary())


if __name__ == "__main__":
    main()
