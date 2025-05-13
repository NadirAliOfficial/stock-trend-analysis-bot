#!/usr/bin/env python3
"""
Milestone 1 – Trend Filter System (IBKR Data)

Requirements:
    pip install ib_insync pandas numpy

Usage:
    python milestone1.py
"""

import pandas as pd
import numpy as np
from ib_insync import IB, Stock, util

# Path to stock symbols CSV
CSV_PATH = 'listStocksAll_SCREENED.csv'


def load_symbols(csv_path=CSV_PATH):
    """Load stock symbols from CSV file."""
    df = pd.read_csv(csv_path)
    return df['stock'].dropna().tolist()


def fetch_historical_ibkr(ib, contract, duration='10 D', bar_size='5 mins'):
    """Fetch historical trade data from IBKR."""
    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr=duration,
        barSizeSetting=bar_size,
        whatToShow='TRADES',
        useRTH=True,
        formatDate=1
    )
    if not bars:
        return None
    df = util.df(bars)
    df['Close'] = df['close']
    return df


def analyze_stock(symbol, ib):
    """Analyze a single stock: EMA, RSI scoring."""
    try:
        contract = Stock(symbol, 'SMART', 'USD')
        ib.qualifyContracts(contract)
        df = fetch_historical_ibkr(ib, contract)
        if df is None or df.empty:
            return {'symbol': symbol, 'status': 'No Data'}

        close = df['Close']
        ema_fast = close.ewm(span=12).mean()
        ema_slow = close.ewm(span=26).mean()

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        score = int((ema_fast.iloc[-1] > ema_slow.iloc[-1]) + (rsi.iloc[-1] > 50))
        return {
            'symbol': symbol,
            'ema_fast': round(ema_fast.iloc[-1], 2),
            'ema_slow': round(ema_slow.iloc[-1], 2),
            'rsi': round(rsi.iloc[-1], 2),
            'score': score,
            'should_trade': score >= 2
        }
    except Exception as e:
        return {'symbol': symbol, 'error': str(e)}


def main():
    ib = IB()
    ib.connect('127.0.0.1', 7497, clientId=1)

    symbols = load_symbols()
    results = [analyze_stock(sym, ib) for sym in symbols]

    df_results = pd.DataFrame(results)
    df_results.sort_values(by='score', ascending=False, inplace=True)
    df_results.to_csv('milestone1_results.csv', index=False)
    print(df_results)


if __name__ == '__main__':
    main()
