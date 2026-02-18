# Oil vs Interest Rate Factor Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import yfinance as yf

plt.style.use("default")


# ------------------------------------------
# 1. Fetch Market Data
# ------------------------------------------

def download_market_data(start="2015-01-01"):

    tickers = {
        "Oil": "CL=F",
        "SP500": "^GSPC"
    }

    data = {}

    for name, ticker in tickers.items():

        df = yf.download(ticker, start=start, progress=False)

        # -------- Robust column handling ----------
        if isinstance(df.columns, pd.MultiIndex):
            if ("Adj Close", ticker) in df.columns:
                series = df[("Adj Close", ticker)]
            else:
                series = df[("Close", ticker)]
        else:
            if "Adj Close" in df.columns:
                series = df["Adj Close"]
            else:
                series = df["Close"]
        # ------------------------------------------

        data[name] = series

    combined = pd.DataFrame(data)
    combined.dropna(inplace=True)

    return combined



# ------------------------------------------
# 2. Load Interest Rate
# ------------------------------------------

def load_interest_rate():

    rate = pd.read_csv("data/interest_rate.csv", index_col=0)
    rate.index = pd.to_datetime(rate.index)

    return rate


# ------------------------------------------
# 3. Merge Datasets
# ------------------------------------------

def merge_data():

    prices = download_market_data()
    rate = load_interest_rate()

    combined = prices.join(rate, how="inner")
    combined.dropna(inplace=True)

    return combined


# ------------------------------------------
# 4. Compute Returns
# ------------------------------------------

def compute_returns(df):

    returns = np.log(df / df.shift(1))
    returns.dropna(inplace=True)

    return returns


# ------------------------------------------
# 5. Visualization
# ------------------------------------------

def plot_relationships(df):

    plt.figure(figsize=(12,6))
    df.corr().plot(kind="bar")
    plt.title("Correlation Overview")
    plt.tight_layout()
    plt.savefig("figures/interest_correlation.png")
    plt.close()


# ------------------------------------------
# 6. Regression Model
# ------------------------------------------

def regression_analysis(returns):

    y = returns["Oil"]
    X = returns[["InterestRate", "SP500"]]

    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    with open("figures/interest_regression.txt","w") as f:
        f.write(model.summary().as_text())

    return model


# ------------------------------------------
# 7. Execution
# ------------------------------------------

def main():

    data = merge_data()
    returns = compute_returns(data)

    plot_relationships(returns)

    model = regression_analysis(returns)

    print(model.summary())


if __name__ == "__main__":
    main()
