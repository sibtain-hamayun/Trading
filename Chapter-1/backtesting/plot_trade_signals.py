# backtesting/plot_trade_signals.py

import matplotlib.pyplot as plt

def plot_trade_signals(df, symbol):
    """
    Plots the stock's closing price with markers for buy and sell signals.
    
    Buy signals (SignalFlag == 1) are marked with green upward arrows,
    and sell signals (SignalFlag == -1) are marked with red downward arrows.
    """
    plt.figure(figsize=(12,4))
    plt.plot(df['date'], df['c_price'], label='Close Price', color='blue')
    
    # Identify buy and sell signal rows.
    buy_signals = df[df['SignalFlag'] == 1]
    sell_signals = df[df['SignalFlag'] == -1]
    
    plt.plot(buy_signals['date'], buy_signals['c_price'], '^', markersize=10, color='green', label='Buy Signal')
    plt.plot(sell_signals['date'], sell_signals['c_price'], 'v', markersize=10, color='red', label='Sell Signal')
    
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'{symbol} - Buy & Sell Signals')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
