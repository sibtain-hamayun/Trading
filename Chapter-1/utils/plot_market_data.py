

import matplotlib.pyplot as plt

def plot_price_data(df):
    """
    Plots the price data along with moving averages.
    If 'c_iv' is available, it is plotted on a secondary y-axis.
    
    Dates on the x-axis are rotated vertically and the figure height is reduced.
    """
    # Reduced height: figsize=(width, height)
    fig, ax1 = plt.subplots(figsize=(12, 4))
    
    # Plot price and moving averages on primary axis.
    ax1.plot(df['date'], df['c_price'], label='Price', color='blue')
    if 'MA_200' in df.columns:
        ax1.plot(df['date'], df['MA_200'], label='MA 200', color='orange')
    if 'MA_50' in df.columns:
        ax1.plot(df['date'], df['MA_50'], label='MA 50', color='green')
    if 'MA_9' in df.columns:
        ax1.plot(df['date'], df['MA_9'], label='MA 9', color='red')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left')
    
    # Plot implied volatility on a secondary axis if available.
    if 'c_iv' in df.columns:
        ax2 = ax1.twinx()
        ax2.plot(df['date'], df['c_iv'], label='Implied Volatility', color='purple', linestyle='--')
        ax2.set_ylabel('Implied Volatility')
        ax2.legend(loc='upper right')
    
    plt.title('Price Data and Moving Averages')
    # Rotate x-axis tick labels vertically
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_technical_indicators(df):
    """
    Plots technical indicators in two subplots:
    - The top subplot displays the MACD indicator.
    - The bottom subplot shows the Stochastic Oscillator.
    
    Dates on the x-axis are rotated vertically and the overall figure height is reduced.
    """
    # Reduced height: figsize=(width, height)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    # Plot MACD
    if 'MACD' in df.columns:
        ax1.plot(df['date'], df['MACD'], label='MACD', color='blue')
        ax1.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax1.set_ylabel('MACD')
        ax1.legend(loc='upper left')
        ax1.set_title('MACD Indicator')
    else:
        ax1.text(0.5, 0.5, 'MACD data not available', horizontalalignment='center')
    
    # Plot Stochastic Oscillator
    if 'stochastic' in df.columns:
        ax2.plot(df['date'], df['stochastic'], label='Stochastic Oscillator', color='red')
        ax2.set_ylabel('Stochastic Oscillator')
        ax2.set_xlabel('Date')
        ax2.legend(loc='upper left')
        ax2.set_title('Stochastic Oscillator')
    else:
        ax2.text(0.5, 0.5, 'Stochastic oscillator data not available', horizontalalignment='center')
    
    # Rotate the x-axis tick labels vertically for the shared x-axis
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    plt.show()
