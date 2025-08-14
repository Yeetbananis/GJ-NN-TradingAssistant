import yfinance as yf
import pandas as pd
import pytz
from pathlib import Path

# Define project root to enable easy file access
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def download_data(ticker="GBPJPY=X", period="2y", interval="5m"):
    """
    Downloads historical data from Yahoo Finance and saves it.

    Args:
        ticker (str): The ticker symbol to download.
        period (str): The period of data to download (e.g., "1y", "2y").
        interval (str): The data interval (e.g., "1m", "5m", "15m").

    Returns:
        pd.DataFrame: DataFrame with historical data.
    """
    print(f"Downloading {interval} data for {ticker} for the last {period}...")
    raw_data_path = PROJECT_ROOT / f"data/raw/{ticker}_{interval}.csv"

    # Download data using yfinance
    df = yf.download(tickers=ticker, period=period, interval=interval)

    if df.empty:
        print("Error: No data downloaded. Check ticker or network connection.")
        return None

    # yfinance returns timezone-aware datetimes (in the exchange's timezone).
    # We will convert to UTC for a standard reference, then to PST for analysis.
    df.index = df.index.tz_convert('UTC')

    # Save the raw data
    raw_data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_data_path)
    print(f"Data saved to {raw_data_path}")

    return df

def load_and_preprocess_data(ticker="GBPJPY=X", interval="5m"):
    """
    Loads raw data, performs basic cleaning, and handles timezones.
    """
    raw_data_path = PROJECT_ROOT / f"data/raw/{ticker}_{interval}.csv"

    if not raw_data_path.exists():
        print("Raw data not found. Downloading first...")
        df = download_data(ticker=ticker, interval=interval)
        if df is None:
            return None
    else:
        print(f"Loading raw data from {raw_data_path}")
        df = pd.read_csv(raw_data_path, index_col='Datetime', parse_dates=True)
        # Ensure the index is timezone-aware (UTC) after loading from CSV
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')

    # Drop rows with any missing values
    df.dropna(inplace=True)

    # Convert column names to lowercase for consistency
    df.columns = [col.lower() for col in df.columns]

    print("Data loaded and preprocessed successfully.")
    return df

if __name__ == '__main__':
    # Example of how to run the data loader
    # This will download and save the data if you run the script directly
    gpbjpy_data = load_and_preprocess_data(ticker="GBPJPY=X", interval="5m")
    if gpbjpy_data is not None:
        print("\nData Head:")
        print(gpbjpy_data.head())
        print("\nData Info:")
        gpbjpy_data.info()