# backtesting/backtest_stocks.py

import pandas as pd
import random
from utils.market_data_analysis import download_tranform

def compute_macd_signal(df, signal_span=9):
    """
    Computes the MACD signal line as the EMA of the MACD.
    Assumes the DataFrame already has a 'MACD' column.
    """
    df = df.copy()
    df['Signal'] = df['MACD'].ewm(span=signal_span, adjust=False).mean()
    return df

def generate_signals(df):
    """
    Generates trading signals based solely on MACD crossovers.
    
    Strategy (long-only):
      - Buy when MACD crosses above its Signal line.
      - Sell when MACD crosses below its Signal line.
    
    Adds the following column:
      - 'SignalFlag': 1 for buy, -1 for sell, 0 otherwise.
    """
    df = df.copy()
    df['SignalFlag'] = 0

    for i in range(1, len(df)):
        # Buy signal: MACD crosses above Signal
        if (df.loc[i-1, 'MACD'] < df.loc[i-1, 'Signal']) and (df.loc[i, 'MACD'] > df.loc[i, 'Signal']):
            df.loc[i, 'SignalFlag'] = 1
        # Sell signal: MACD crosses below Signal
        elif (df.loc[i-1, 'MACD'] > df.loc[i-1, 'Signal']) and (df.loc[i, 'MACD'] < df.loc[i, 'Signal']):
            df.loc[i, 'SignalFlag'] = -1
        else:
            df.loc[i, 'SignalFlag'] = 0

    return df

def prepare_backtest_data(stock_symbol='AAPL', data_period='1y'):
    """
    Prepares data for backtesting:
      1. Downloads and processes market data using your existing download_tranform().
      2. Computes the MACD signal line.
      3. Generates buy/sell signals based solely on MACD.
      
    Returns a DataFrame ready for simulation.
    """
    df = download_tranform(stock_symbol, data_period)
    df = compute_macd_signal(df)
    df = generate_signals(df)
    return df

def simulate_macd_strategy(df, initial_capital=100000):
    """
    Simulates trades based on the MACD strategy.
    Uses a fixed budget and goes all-in when a buy signal occurs and sells all on a sell signal.
    
    Returns:
      - Updated DataFrame with columns: 'PortfolioValue', 'Cash', 'Shares'.
      - A trade log DataFrame with trade details.
    """
    cash = initial_capital
    shares = 0
    buy_price = None

    portfolio_values = []
    cash_history = []
    shares_history = []
    trade_log = []

    for index, row in df.iterrows():
        current_price = row['c_price']
        portfolio_value = cash + shares * current_price
        portfolio_values.append(portfolio_value)
        cash_history.append(cash)
        shares_history.append(shares)

        signal = row['SignalFlag']
        if signal == 1 and shares == 0:
            # Buy all-in: purchase as many shares as possible.
            shares_to_buy = int(cash // current_price)
            if shares_to_buy > 0:
                buy_price = current_price
                cash -= shares_to_buy * current_price
                shares = shares_to_buy
                trade_log.append({
                    'Date': row['date'],
                    'Action': 'Buy',
                    'Shares': shares_to_buy,
                    'Price': current_price,
                    'Signal': signal,
                    'PnL': None
                })
        elif signal == -1 and shares > 0:
            sell_price = current_price
            pnl = (sell_price - buy_price) * shares
            cash += shares * sell_price
            trade_log.append({
                'Date': row['date'],
                'Action': 'Sell',
                'Shares': shares,
                'Price': sell_price,
                'Signal': signal,
                'PnL': pnl
            })
            shares = 0
            buy_price = None

    df['PortfolioValue'] = portfolio_values
    df['Cash'] = cash_history
    df['Shares'] = shares_history
    trade_df = pd.DataFrame(trade_log)
    return df, trade_df

def backtest_buy_and_hold(df, initial_capital=100000):
    """
    Simulates a simple buy-and-hold strategy.
    
    Assumes the entire budget is invested on the first day.
    Returns an updated DataFrame with 'PortfolioValueBH'.
    For compatibility in comparisons, it also copies this column to 'PortfolioValue'.
    """
    df = df.copy()
    df['MarketReturn'] = df['c_price'].pct_change()
    df['PortfolioValueBH'] = (1 + df['MarketReturn']).cumprod() * initial_capital
    # Set PortfolioValue equal to the buy-and-hold portfolio value for consistent plotting.
    df['PortfolioValue'] = df['PortfolioValueBH']
    return df

def simulate_random_strategy(df, initial_capital=100000, chunk_size=5000):
    """
    Simulates a random strategy with fixed monetary chunks.
    
    At each iteration, a random decision is made:
      - 60% chance: No action.
      - 20% chance: Execute a Buy chunk.
      - 20% chance: Execute a Sell chunk.
    
    Rules:
      - Buy: If cash >= chunk_size, purchase shares worth 'chunk_size' at the current price.
      - Sell: If shares are held and current price > average cost,
          sell shares worth 'chunk_size'.  
          Otherwise (if current price <= average cost), do not sell; instead, buy another chunk to lower the average cost.
    
    Maintains:
      - cash: Available cash.
      - shares: Number of shares held.
      - avg_cost: Average purchase price.
    
    Returns:
      - Updated DataFrame with 'PortfolioValue', 'Cash', 'Shares', 'AvgCost'.
      - A trade log DataFrame with trade details.
    """
    cash = initial_capital
    shares = 0
    avg_cost = None

    portfolio_values = []
    cash_history = []
    shares_history = []
    avg_cost_history = []
    trade_log = []

    for index, row in df.iterrows():
        current_price = row['c_price']
        portfolio_value = cash + shares * current_price
        portfolio_values.append(portfolio_value)
        cash_history.append(cash)
        shares_history.append(shares)
        avg_cost_history.append(avg_cost if avg_cost is not None else 0)

        r = random.random()
        if r < 0.6:
            action = "None"
        elif r < 0.8:
            action = "Buy"
        else:
            action = "Sell"

        if action == "Buy":
            if cash >= chunk_size:
                shares_to_buy = int(chunk_size // current_price)
                if shares_to_buy > 0:
                    if shares == 0:
                        avg_cost = current_price
                    else:
                        avg_cost = (shares * avg_cost + shares_to_buy * current_price) / (shares + shares_to_buy)
                    cash -= shares_to_buy * current_price
                    shares += shares_to_buy
                    trade_log.append({
                        'Date': row['date'],
                        'Action': 'Buy',
                        'Shares': shares_to_buy,
                        'Price': current_price,
                        'Chunk': chunk_size,
                        'NewAvgCost': avg_cost,
                        'Signal': action
                    })
        elif action == "Sell":
            if shares > 0:
                if current_price > avg_cost:
                    shares_to_sell = int(chunk_size // current_price)
                    if shares_to_sell > shares:
                        shares_to_sell = shares
                    if shares_to_sell > 0:
                        cash += shares_to_sell * current_price
                        pnl = (current_price - avg_cost) * shares_to_sell
                        trade_log.append({
                            'Date': row['date'],
                            'Action': 'Sell',
                            'Shares': shares_to_sell,
                            'Price': current_price,
                            'Chunk': chunk_size,
                            'PnL': pnl,
                            'Signal': action
                        })
                        shares -= shares_to_sell
                        if shares == 0:
                            avg_cost = None
                else:
                    # Do not sell at a loss; instead, if cash is available, buy another chunk to average down.
                    if cash >= chunk_size:
                        shares_to_buy = int(chunk_size // current_price)
                        if shares_to_buy > 0:
                            if shares == 0:
                                avg_cost = current_price
                            else:
                                avg_cost = (shares * avg_cost + shares_to_buy * current_price) / (shares + shares_to_buy)
                            cash -= shares_to_buy * current_price
                            shares += shares_to_buy
                            trade_log.append({
                                'Date': row['date'],
                                'Action': 'Buy (Averaging Down)',
                                'Shares': shares_to_buy,
                                'Price': current_price,
                                'Chunk': chunk_size,
                                'NewAvgCost': avg_cost,
                                'Signal': action
                            })
        # No action if "None".
    df['PortfolioValue'] = portfolio_values
    df['Cash'] = cash_history
    df['Shares'] = shares_history
    df['AvgCost'] = avg_cost_history
    trade_df = pd.DataFrame(trade_log)
    return df, trade_df
