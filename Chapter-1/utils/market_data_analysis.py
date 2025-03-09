# market_analysis.py

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm
from ib_insync import IB, Stock
import random
import nest_asyncio

# Enable nested asyncio to allow IBKR connections in notebooks or interactive sessions.
nest_asyncio.apply()

# -----------------------------
# IBKR Volatility Data Function
# -----------------------------
def fetch_iv_data(ib, contract, endDateTime='', durationStr='3 Y',
                  barSizeSetting='1 day', whatToShow='OPTION_IMPLIED_VOLATILITY',
                  useRTH=True, formatDate=1):
    """
    Fetches implied volatility data from IBKR for a given contract.
    
    Parameters:
        ib: IB instance (from ib_insync).
        contract: The contract to request data for.
        endDateTime, durationStr, barSizeSetting, whatToShow, useRTH, formatDate:
            Parameters passed to IB's reqHistoricalData.
    
    Returns:
        A pandas DataFrame with 'date' and 'c_iv' (implied volatility) columns.
    """
    iv_data = ib.reqHistoricalData(
        contract,
        endDateTime=endDateTime,
        durationStr=durationStr,
        barSizeSetting=barSizeSetting,
        whatToShow=whatToShow,
        useRTH=useRTH,
        formatDate=formatDate
    )
    return pd.DataFrame([{'date': bar.date, 'c_iv': bar.close} for bar in iv_data])

# -----------------------------
# IBKR Connection Function
# -----------------------------
def connect_ibkr(client_id=None, host='127.0.0.1', port=7496):
    """
    Establishes a connection to the IBKR TWS or Gateway.
    
    Parameters:
        client_id: Optional client ID. If not provided, a random ID is used.
        host: The host address for IBKR.
        port: The port number for IBKR.
    
    Returns:
        An instance of IB that is connected.
    """
    ib = IB()
    if client_id is None:
        client_id = random.randint(1, 9999)
    ib.connect(host, port, clientId=client_id)
    return ib

# -----------------------------
# Price Data Download Function
# -----------------------------
def download_price_data(stock_symbol, data_period='3y', interval='1d'):
    """
    Downloads historical price data using yfinance.
    
    Parameters:
        stock_symbol: Ticker symbol for the stock.
        data_period: Data period (e.g., '3y' for three years).
        interval: Data interval (e.g., '1d' for daily data).
    
    Returns:
        A pandas DataFrame with at least 'date' and 'c_price' (close price) columns.
    """
    price_df = yf.download(stock_symbol, period=data_period, interval=interval)
    price_df.reset_index(inplace=True)
    if isinstance(price_df.columns, pd.MultiIndex):
        price_df.columns = price_df.columns.get_level_values(0)
    # Ensure column names are consistent
    price_df.rename(columns={'Date': 'date', 'Close': 'c_price'}, inplace=True)
    price_df['date'] = pd.to_datetime(price_df['date'])
    return price_df

# -----------------------------
# Data Merge Function
# -----------------------------
def merge_data(price_df, iv_df):
    """
    Merges price data and volatility data on the date column.
    
    Parameters:
        price_df: DataFrame containing price data.
        iv_df: DataFrame containing implied volatility data.
    
    Returns:
        A merged DataFrame sorted by date.
    """
    merged_df = pd.merge(price_df[['date', 'c_price']], iv_df, on='date', how='inner')
    merged_df.sort_values(by='date', inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    return merged_df

# -----------------------------
# Technical Indicators Calculation
# -----------------------------
def compute_technical_indicators(merged_df):
    """
    Computes moving averages, gradients, stochastic oscillator, and MACD.
    
    Parameters:
        merged_df: DataFrame containing merged price and IV data.
    
    Returns:
        The DataFrame with additional technical indicator columns.
    """
    # Moving averages
    merged_df['MA_200'] = merged_df['c_price'].rolling(window=200).mean()
    merged_df['MA_50'] = merged_df['c_price'].rolling(window=50).mean()
    merged_df['MA_9'] = merged_df['c_price'].rolling(window=9).mean()

    # Gradients for moving averages
    merged_df['MA_200_grad'] = merged_df['MA_200'].diff()
    merged_df['MA_50_grad'] = merged_df['MA_50'].diff()
    merged_df['MA_9_grad'] = merged_df['MA_9'].diff()

    def gradient_direction(grad):
        if pd.isna(grad):
            return None
        elif grad > 0:
            return 'increasing'
        elif grad < 0:
            return 'decreasing'
        else:
            return 'flat'

    merged_df['MA_200_dir'] = merged_df['MA_200_grad'].apply(gradient_direction)
    merged_df['MA_50_dir'] = merged_df['MA_50_grad'].apply(gradient_direction)
    merged_df['MA_9_dir'] = merged_df['MA_9_grad'].apply(gradient_direction)

    # Stochastic Oscillator Calculation (10, 3, 3)
    merged_df['lowest_10'] = merged_df['c_price'].rolling(window=10).min()
    merged_df['highest_10'] = merged_df['c_price'].rolling(window=10).max()
    merged_df['stochastic_%K'] = 100 * (merged_df['c_price'] - merged_df['lowest_10']) / (merged_df['highest_10'] - merged_df['lowest_10'])
    merged_df['stochastic_slow_%K'] = merged_df['stochastic_%K'].rolling(window=3).mean()
    merged_df['stochastic_%D'] = merged_df['stochastic_slow_%K'].rolling(window=3).mean()
    # Use the smoothed %K as the oscillator value.
    merged_df['stochastic'] = merged_df['stochastic_slow_%K']

    # MACD Calculation: EMA(12) - EMA(26)
    merged_df['EMA_12'] = merged_df['c_price'].ewm(span=12, adjust=False).mean()
    merged_df['EMA_26'] = merged_df['c_price'].ewm(span=26, adjust=False).mean()
    merged_df['MACD'] = merged_df['EMA_12'] - merged_df['EMA_26']

    # Drop intermediate columns
    merged_df.drop(columns=['lowest_10', 'highest_10', 'stochastic_%K', 'stochastic_slow_%K'], inplace=True)
    
    return merged_df

# -----------------------------
# Main Function to Execute Workflow
# -----------------------------
def download_tranform(stock_symbol='QQQ', data_period='3y'):
    """
    Main function to perform the entire workflow:
    1. Connect to IBKR and fetch IV data.
    2. Download price data.
    3. Merge the datasets.
    4. Compute technical indicators.
    
    Parameters:
        stock_symbol: The ticker symbol to use.
        data_period: The period over which to download price data.
    
    Returns:
        The final merged DataFrame with computed technical indicators.
    """
    print(f"Starting download and transformation process for {stock_symbol} with data period {data_period}...")
    
    # Connect to IBKR
    print("Connecting to IBKR...")
    ib = connect_ibkr()
    print("IBKR connection established.")
    
    stock_contract = Stock(stock_symbol, 'SMART', 'USD')
    print(f"Created stock contract for {stock_symbol}.")
    
    # Fetch IV data from IBKR
    print("Fetching IV data from IBKR...")
    iv_df = fetch_iv_data(ib, stock_contract)
    iv_df['date'] = pd.to_datetime(iv_df['date'])
    print(f"IV data fetched: {iv_df.shape[0]} records retrieved.")
    
    # Download price data from Yahoo Finance
    print("Downloading price data from Yahoo Finance...")
    price_df = download_price_data(stock_symbol, data_period=data_period)
    print(f"Price data downloaded: {price_df.shape[0]} records retrieved.")
    
    # Merge data
    print("Merging price and IV data...")
    merged_df = merge_data(price_df, iv_df)
    print(f"Data merged: {merged_df.shape[0]} records in the merged dataset.")
    
    # Compute technical indicators
    print("Computing technical indicators...")
    merged_df = compute_technical_indicators(merged_df)
    print("Technical indicators computed successfully.")
    
    # Disconnect from IBKR to clean up connection
    print("Disconnecting from IBKR...")
    ib.disconnect()
    print("Disconnected from IBKR.")
    
    print("Download and transformation process completed.")
    return merged_df

# -----------------------------
# Run as Script Example
# -----------------------------
if __name__ == '__main__':
    df = download_tranform()
    print("Final merged data shape:", df.shape)
