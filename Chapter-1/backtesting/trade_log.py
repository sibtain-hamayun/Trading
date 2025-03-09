# backtesting/trade_log.py

import pandas as pd

def generate_trade_log(df):
    """
    Generates a trade log DataFrame from the backtested DataFrame.
    
    The input DataFrame `df` is expected to have:
      - 'date': The date of the observation.
      - 'c_price': The closing price.
      - 'SignalFlag': 1 for a buy signal, -1 for a sell signal, 0 otherwise.
    
    The output DataFrame contains the following columns:
      - 'Date': The date of the trade event.
      - 'Action': 'Buy' or 'Sell'.
      - 'Price': The price at which the trade occurred.
      - 'SignalFlag': The signal value that triggered the action.
      - 'PnL': The profit and loss for a sell event (None for buy events).
    
    The function assumes a long-only strategy:
      - A trade is entered on a buy signal (SignalFlag == 1).
      - It is closed on the subsequent sell signal (SignalFlag == -1).
      - The PnL is computed as (exit price - entry price) for each closed trade.
    """
    trades = []
    entry_price = None
    entry_date = None

    # Iterate through the DataFrame row by row.
    for index, row in df.iterrows():
        signal = row['SignalFlag']
        
        # If a buy signal is encountered and there is no open trade, record the buy.
        if signal == 1 and entry_price is None:
            entry_price = row['c_price']
            entry_date = row['date']
            trades.append({
                'Date': row['date'],
                'Action': 'Buy',
                'Price': row['c_price'],
                'SignalFlag': signal,
                'PnL': None
            })
        # If a sell signal is encountered and a trade is open, record the sell and compute PnL.
        elif signal == -1 and entry_price is not None:
            exit_price = row['c_price']
            pnl = exit_price - entry_price  # absolute profit/loss per share
            trades.append({
                'Date': row['date'],
                'Action': 'Sell',
                'Price': row['c_price'],
                'SignalFlag': signal,
                'PnL': pnl
            })
            # Reset for the next trade.
            entry_price = None
            entry_date = None

    trade_df = pd.DataFrame(trades)
    return trade_df
