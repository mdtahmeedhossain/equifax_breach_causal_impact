"""
Data Collection Module for Equifax Breach Analysis
Fetches stock price data and calculates returns
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

def fetch_stock_data(tickers, start_date, end_date):
    """
    Fetch adjusted closing prices for multiple tickers using yfinance.
    
    Parameters:
    -----------
    tickers : list
        List of stock tickers
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: date, ticker, price, returns
    """
    all_data = []
    
    for ticker in tickers:
        try:
            # Fetch data using yfinance
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                print(f"Warning: No data found for {ticker}")
                continue
            
            # Extract adjusted close prices
            df = df[['Close']].reset_index()
            df.columns = ['date', 'price']
            df['ticker'] = ticker
            
            # Calculate daily returns
            df['returns'] = df['price'].pct_change()
            
            all_data.append(df)
            print(f"✓ Fetched {len(df)} days for {ticker}")
            
        except Exception as e:
            print(f"✗ Error fetching {ticker}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No data could be fetched for any ticker")
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Remove NaN returns (first day for each ticker)
    combined_df = combined_df.dropna(subset=['returns'])
    
    return combined_df


def prepare_analysis_data(df, event_date, treated_ticker, window_days=180, window_days_after=21):
    """
    Prepare data for DiD and SCM analysis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw stock data with columns: date, ticker, price, returns
    event_date : datetime
        Date of the event (treatment)
    treated_ticker : str
        Ticker symbol of the treated unit
    window_days : int
        Number of days before event to include
    window_days_after : int
        Number of days after event to include
    
    Returns:
    --------
    pd.DataFrame
        Prepared data with treatment indicators
    """
    # Convert event_date to datetime if string
    if isinstance(event_date, str):
        event_date = pd.to_datetime(event_date)
    
    # Make dates timezone-aware if needed
    if df['date'].dt.tz is not None:
        if event_date.tz is None:
            event_date = event_date.tz_localize('America/New_York')
    else:
        if event_date.tz is not None:
            event_date = event_date.tz_localize(None)
    
    # Define time window (now includes post-treatment period)
    start_date = event_date - timedelta(days=window_days)
    end_date = event_date + timedelta(days=window_days_after)
    
    # Filter data to include both pre and post treatment
    df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    
    # Create treatment indicators
    df_filtered['treated'] = (df_filtered['ticker'] == treated_ticker).astype(int)
    df_filtered['post'] = (df_filtered['date'] >= event_date).astype(int)
    df_filtered['treated_post'] = df_filtered['treated'] * df_filtered['post']
    
    # Calculate days relative to event
    df_filtered['days_to_event'] = (df_filtered['date'] - event_date).dt.days
    
    return df_filtered


if __name__ == "__main__":
    # Test the module
    tickers = ['EFX', 'MCO', 'TRU', 'SPY', 'VTI', 'EXPGY', 'BAH']
    start_date = '2017-01-01'
    end_date = '2017-09-30'
    event_date = '2017-09-08'
    
    print("Fetching stock data...")
    df = fetch_stock_data(tickers, start_date, end_date)
    print(f"\nTotal records: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Tickers: {df['ticker'].unique().tolist()}")
    
    print("\nPreparing analysis data...")
    df_prepared = prepare_analysis_data(df, event_date, 'EFX')
    print(f"Analysis records: {len(df_prepared)}")
    print(f"Pre-treatment: {df_prepared[df_prepared['post']==0].shape[0]}")
    print(f"Post-treatment: {df_prepared[df_prepared['post']==1].shape[0]}")
