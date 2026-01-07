"""Data collection and preparation for Equifax breach analysis."""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

def fetch_stock_data(tickers, start_date, end_date):
    """Fetch stock prices and calculate returns."""
    all_data = []

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)

            if df.empty:
                print(f"Warning: No data for {ticker}")
                continue

            df = df[['Close']].reset_index()
            df.columns = ['date', 'price']
            df['ticker'] = ticker
            df['returns'] = df['price'].pct_change()

            all_data.append(df)
            print(f"Fetched {ticker}: {len(df)} days")

        except Exception as e:
            print(f"Error: {ticker} - {e}")
            continue

    if not all_data:
        raise ValueError("No data could be fetched for any ticker")

    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.dropna(subset=['returns'])

    return combined_df


def prepare_analysis_data(df, event_date, treated_ticker, window_days=180, window_days_after=21):
    """Prepare data with treatment indicators for causal analysis."""
    if isinstance(event_date, str):
        event_date = pd.to_datetime(event_date)

    # Handle timezone
    if df['date'].dt.tz is not None:
        if event_date.tz is None:
            event_date = event_date.tz_localize('America/New_York')
    else:
        if event_date.tz is not None:
            event_date = event_date.tz_localize(None)

    start_date = event_date - timedelta(days=window_days)
    end_date = event_date + timedelta(days=window_days_after)

    df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()

    df_filtered['treated'] = (df_filtered['ticker'] == treated_ticker).astype(int)
    df_filtered['post'] = (df_filtered['date'] >= event_date).astype(int)
    df_filtered['treated_post'] = df_filtered['treated'] * df_filtered['post']
    df_filtered['days_to_event'] = (df_filtered['date'] - event_date).dt.days

    return df_filtered


if __name__ == "__main__":
    tickers = ['EFX', 'MCO', 'TRU', 'SPY', 'VTI', 'EXPGY', 'BAH']
    start_date = '2017-01-01'
    end_date = '2017-09-30'
    event_date = '2017-09-08'

    df = fetch_stock_data(tickers, start_date, end_date)
    print(f"Total: {len(df)} records")
    print(f"Range: {df['date'].min()} to {df['date'].max()}")

    df_prepared = prepare_analysis_data(df, event_date, 'EFX')
    print(f"Pre: {df_prepared[df_prepared['post']==0].shape[0]}, Post: {df_prepared[df_prepared['post']==1].shape[0]}")
