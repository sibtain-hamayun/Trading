# backtesting/plot_backtest.py

import matplotlib.pyplot as plt

def plot_portfolio_comparison(dfs, labels, symbol):
    """
    Plots the portfolio value curves of multiple strategies on a single figure.
    
    Parameters:
      dfs: List of DataFrames, each containing a 'PortfolioValue' column.
      labels: List of labels corresponding to each strategy.
      symbol: Stock symbol for the title.
    """
    plt.figure(figsize=(12,4))
    for df, label in zip(dfs, labels):
        plt.plot(df['date'], df['PortfolioValue'], label=label)
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.title(f'Portfolio Value Comparison for {symbol}')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_portfolio_components(df, symbol):
    """
    Plots cash held and shares held on two separate subplots within the same figure.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,6), sharex=True)
    
    ax1.plot(df['date'], df['Cash'], label='Cash Held', color='green')
    ax1.set_ylabel('Cash Held')
    ax1.legend(loc='upper left')
    
    ax2.plot(df['date'], df['Shares'], label='Shares Held', color='blue')
    ax2.set_ylabel('Shares Held')
    ax2.set_xlabel('Date')
    ax2.legend(loc='upper left')
    
    plt.suptitle(f'{symbol} - Portfolio Components Over Time')
    plt.xticks(rotation=45)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
