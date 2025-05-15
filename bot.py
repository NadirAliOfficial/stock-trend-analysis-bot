
import pandas as pd
import numpy as np
from ib_insync import IB, Stock, util
import time

# Configuration
CSV_PATH      = 'listStocksAll_SCREENED.csv'
RESULTS_PATH  = 'milestone1_results.csv'
IB_HOST       = '127.0.0.1'
IB_PORT       = 7497
CLIENT_ID     = 1
USE_RTH       = True
SLEEP_SECONDS = 1  # throttle between requests

# Define timeframes for momentum analysis: (duration, bar size)
TIMEFRAMES = [
    ('5 Y', '1 day'),
    ('1 Y', '1 day'),
    ('6 M', '1 day'),
    ('1 M', '1 hour'),
    ('1 W', '5 mins'),
    ('1 D', '5 mins'),
]

def load_symbols(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)
    return df['stock'].dropna().tolist()

def fetch_historical_ibkr(ib, contract, duration, bar_size):
    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr=duration,
        barSizeSetting=bar_size,
        whatToShow='TRADES',
        useRTH=USE_RTH,
        formatDate=1
    )
    if not bars:
        return None
    return util.df(bars)

def analyze_stock(symbol, ib):
    result = {'symbol': symbol}
    try:
        contract = Stock(symbol, 'SMART', 'USD')
        ib.qualifyContracts(contract)

        # Primary data for EMA/RSI
        df_main = fetch_historical_ibkr(ib, contract, '10 D', '5 mins')
        if df_main is None or df_main.empty:
            result['status'] = 'No Data'
            return result

        close = df_main['close']
        ema_fast = close.ewm(span=12).mean()
        ema_slow = close.ewm(span=26).mean()
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # Multi-timeframe momentum
        returns = []
        for dur, bs in TIMEFRAMES:
            df_tf = fetch_historical_ibkr(ib, contract, dur, bs)
            if df_tf is None or df_tf.empty:
                continue
            first, last = df_tf['close'].iloc[0], df_tf['close'].iloc[-1]
            returns.append((last - first) / first * 100)
            time.sleep(SLEEP_SECONDS)

        momentum_hits = sum(1 for r in returns if r > 0)
        momentum_flag = momentum_hits >= 4  # need ≥4 positive segments

        # Final scoring
        score = 0
        if ema_fast.iloc[-1] > ema_slow.iloc[-1]:
            score += 1
        if rsi.iloc[-1] > 50:
            score += 1
        if momentum_flag:
            score += 1

        result.update({
            'ema_fast': round(ema_fast.iloc[-1], 2),
            'ema_slow': round(ema_slow.iloc[-1], 2),
            'rsi': round(rsi.iloc[-1], 2),
            'momentum_hits': momentum_hits,
            'score': score,
            'should_trade': score >= 2
        })
        return result

    except Exception as e:
        result['error'] = str(e)
        return result

def main():
    ib = IB()
    ib.connect(IB_HOST, IB_PORT, clientId=CLIENT_ID)

    symbols = load_symbols()
    results = []
    for sym in symbols:
        print(f"Processing {sym}")
        res = analyze_stock(sym, ib)
        results.append(res)
        time.sleep(SLEEP_SECONDS)

    ib.disconnect()

    df = pd.DataFrame(results)
    df.sort_values(by='score', ascending=False, inplace=True)
    df.to_csv(RESULTS_PATH, index=False)
    print(df)

if __name__ == '__main__':
    main()
